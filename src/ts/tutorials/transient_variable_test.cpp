static char help[] =
    "Test to check the PETSc machinery for the TS \n"
    "Explicit (RK):  ./ts_example -ts_monitor_error -ts_dt 0.1 -ts_type rk -ts_adapt_type none\n"
    "Implicit (BDF): ./ts_example -ts_monitor_error -ts_dt 0.1 -ts_type bdf -ts_bdf_order 2 "
    "-ts_adapt_type none -condition_system -complex_fd -pc_type none\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmshell.h>
#include <petscts.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <vector>

// ==============================================================================
// C++ View Wrappers
// ==============================================================================
template <std::floating_point T>
struct VecSpan
{
  private:
    Vec          m_vec;
    std::span<T> m_span;

    static constexpr bool is_const = std::is_const_v<T>;
    using RawT                     = std::remove_const_t<T>;

  public:
    VecSpan()                          = delete;
    VecSpan(const VecSpan&)            = delete;
    VecSpan& operator=(const VecSpan&) = delete;

    explicit VecSpan(Vec v) : m_vec(v)
    {
        PetscInt size = 0;
        PetscCallVoid(VecGetLocalSize(m_vec, &size));

        if constexpr (is_const)
        {
            const RawT* ptr = nullptr;
            PetscCallVoid(VecGetArrayRead(m_vec, &ptr));
            m_span = {ptr, static_cast<std::size_t>(size)};
        }
        else
        {
            RawT* ptr = nullptr;
            PetscCallVoid(VecGetArray(m_vec, &ptr));
            m_span = {ptr, static_cast<std::size_t>(size)};
        }
    }

    ~VecSpan()
    {
        if constexpr (is_const)
        {
            const RawT* ptr = m_span.data();
            PetscCallVoid(VecRestoreArrayRead(m_vec, &ptr));
        }
        else
        {
            RawT* ptr = m_span.data();
            PetscCallVoid(VecRestoreArray(m_vec, &ptr));
        }
    }

    std::span<T> span() { return m_span; }
    T&           operator[](std::size_t i) { return m_span[i]; }
    std::size_t  size() const { return m_span.size(); }
};

using VecReadSpan  = VecSpan<const PetscReal>;
using VecWriteSpan = VecSpan<PetscReal>;
using CplxType     = std::complex<PetscReal>;

// ==============================================================================
// Application Context
// ==============================================================================
struct AppCtx
{
    bool      ts_implicit{false};
    bool      condition_system{false};
    bool      complex_fd{false};
    PetscReal shift{};
    PetscReal epsilon{1e-30};

    Vec                   u_curr{};
    std::vector<CplxType> u_plus_iv{};
    std::vector<CplxType> complex_fun_eval{};
};

enum class TimeInt
{
    Implicit = 0,
    Explicit
};

// ==============================================================================
// Math & Physics Implementations
// ==============================================================================
template <typename T>
void tilde_to_u_ker(std::span<const T> u_tilde, std::span<T> u)
{
    std::ranges::transform(u_tilde, u.begin(), [](auto u_ti) { return std::exp(u_ti); });
    return;
}
template <typename T>
void u_to_tilde_ker(std::span<const T> u, std::span<T> u_tilde)
{
    std::ranges::transform(u, u_tilde.begin(), [](auto u_i) { return std::log(u_i); });
    return;
}

PetscErrorCode tilde_to_u(const Vec u_tilde, Vec u)
{
    PetscFunctionBeginUser;
    VecReadSpan  u_t_sp(u_tilde);
    VecWriteSpan u_sp(u);
    tilde_to_u_ker<PetscReal>(u_t_sp.span(), u_sp.span());
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode u_to_tilde(const Vec u, Vec u_tilde)
{
    PetscFunctionBeginUser;
    VecWriteSpan u_t_sp(u_tilde);
    VecReadSpan  u_sp(u);
    u_to_tilde_ker<PetscReal>(u_sp.span(), u_t_sp.span());
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode primitive_to_conservative(TS ts, Vec in, Vec out, void* ctx)
{
    PetscFunctionBeginUser;
    PetscCall(tilde_to_u(in, out));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T, TimeInt time_int>
void my_rhs_impl(std::span<const T> u, std::span<T> f)
{
    T c_type{};
    if constexpr (time_int == TimeInt::Explicit) c_type = T{-1.0};
    if constexpr (time_int == TimeInt::Implicit) c_type = T{1.0};

    f[0] = c_type * (u[0] - u[1]);
    f[1] = c_type * (u[1] - u[2]);
    f[2] = c_type * (u[2]);
}

PetscErrorCode rhs_func(TS ts, PetscReal t, Vec v_in, Vec v_out, void* ctx)
{
    PetscFunctionBeginUser;
    VecReadSpan  u_view{v_in};
    VecWriteSpan g_view{v_out};
    my_rhs_impl<PetscReal, TimeInt::Explicit>(u_view.span(), g_view.span());
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode petsc_sol_fun(TS ts, PetscReal t, Vec u, void* ctx)
{
    PetscFunctionBeginUser;
    const auto   et = std::exp(-t);
    VecWriteSpan u_view{u};
    auto         u_span = u_view.span();
    u_span[0]           = (1.0 + t + 0.5 * t * t) * et;
    u_span[1]           = (1.0 + t) * et;
    u_span[2]           = et;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode petsc_sol_fun_tilde_var(TS ts, PetscReal t, Vec u, void* ctx)
{
    PetscFunctionBeginUser;
    PetscCall(petsc_sol_fun(ts, t, u, ctx));
    PetscCall(u_to_tilde(u, u));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <bool NeedsChangeOfVariable>
PetscErrorCode comp_step_deriv(Mat J, Vec v, Vec w)
{
    PetscFunctionBeginUser;
    AppCtx* ctx = nullptr;
    PetscCall(MatShellGetContext(J, &ctx));

    VecReadSpan  v_view{v};
    VecReadSpan  u_view{ctx->u_curr};
    VecWriteSpan jv_res{w};

    std::span<CplxType> u_pl_iv{ctx->u_plus_iv};
    std::span<CplxType> f_u_iv{ctx->complex_fun_eval};

    const auto sigma      = ctx->shift;
    const auto h          = ctx->epsilon;
    const auto one_over_h = 1.0 / h;

    auto u_span  = u_view.span();
    auto v_span  = v_view.span();
    auto jv_span = jv_res.span();

    for (auto&& [u, v_i, cplx_out] : std::views::zip(u_span, v_span, u_pl_iv))
        cplx_out = std::complex<PetscReal>{u, v_i * h};

    if constexpr (NeedsChangeOfVariable) tilde_to_u_ker<CplxType>(u_pl_iv, u_pl_iv);

    my_rhs_impl<CplxType, TimeInt::Explicit>(u_pl_iv, f_u_iv);

    for (auto&& [f_cplx, u_iv, jv_out] : std::views::zip(f_u_iv, u_pl_iv, jv_span))
    {
        jv_out = std::imag(f_cplx + sigma * u_iv) * one_over_h;
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <bool NeedsChangeOfVariable>
static PetscErrorCode pre_jv_eval(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal shift, Mat J,
                                  Mat P, void* ctx_void)
{
    PetscFunctionBeginUser;
    auto ctx = static_cast<AppCtx*>(ctx_void);
    PetscCall(VecCopy(U, ctx->u_curr));
    ctx->shift = shift;
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <bool NeedsChangeOfVariable>
static PetscErrorCode fun_implicit(TS ts, PetscReal t, Vec U, Vec U_t, Vec F, void* ctx_void)
{
    PetscFunctionBeginUser;
    auto         ctx = static_cast<AppCtx*>(ctx_void);
    VecWriteSpan f_view{F};

    if constexpr (NeedsChangeOfVariable)
    {
        PetscCall(tilde_to_u(U, ctx->u_curr));
        VecReadSpan u_view{ctx->u_curr};
        my_rhs_impl<PetscReal, TimeInt::Implicit>(u_view.span(), f_view.span());
    }
    else
    {
        VecReadSpan u_view{U};
        my_rhs_impl<PetscReal, TimeInt::Implicit>(u_view.span(), f_view.span());
    }

    PetscCall(VecAXPY(F, 1.0, U_t));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ==============================================================================
// Main Routine
// ==============================================================================

int main(int argc, char** argv)
{
    TS       ts;
    Vec      u_sol;
    Mat      j_shell = NULL;
    AppCtx   ctx;
    PetscInt prob_size = 3;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscCall(PetscOptionsGetBool(NULL, NULL, "-condition_system", &ctx.condition_system, NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-complex_fd", &ctx.complex_fd, NULL));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Complex step deriv : %s\n",
                          ctx.complex_fd ? "true" : "false"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Change of variable : %s\n",
                          ctx.condition_system ? "true" : "false"));

    // Vectors setup
    PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, prob_size, &u_sol));
    PetscCall(VecSet(u_sol, 1.0));
    PetscCall(VecDuplicate(u_sol, &ctx.u_curr));
    ctx.u_plus_iv.resize(prob_size);
    ctx.complex_fun_eval.resize(prob_size);

    // TS setup
    PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
    PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
    PetscCall(TSSetTimeStep(ts, 0.01));
    PetscCall(TSSetMaxTime(ts, 0.1));
    PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
    PetscCall(TSSetSolutionFunction(ts, petsc_sol_fun, &ctx));

    // Allow CLI to override type before we check explicit vs implicit
    PetscCall(TSSetFromOptions(ts));

    TSType ts_type;
    PetscCall(TSGetType(ts, &ts_type));
    std::string_view type_str(ts_type);

    if (type_str == TSBDF || type_str == TSTHETA || type_str == TSALPHA || type_str == TSBEULER ||
        type_str == TSCN || type_str == TSARKIMEX || type_str == TSROSW)
    {
        ctx.ts_implicit = true;
    }

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Time int implicit  : %s\n",
                          ctx.ts_implicit ? "true" : "false"));

    // Handle Time Integration scheme specifics
    if (ctx.ts_implicit)
    {
        PetscCall(TSSetIFunction(
            ts, NULL, ctx.condition_system ? fun_implicit<true> : fun_implicit<false>, &ctx));

        DM dm;
        PetscCall(TSGetDM(ts, &dm));
        PetscCall(DMShellSetGlobalVector(dm, u_sol));

        if (ctx.complex_fd)
        {
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                                  "================================\n Complex FD "
                                  "activated.\n================================\n"));
            PetscCall(MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, prob_size,
                                     prob_size, &ctx, &j_shell));
            if (!ctx.condition_system)
            {
                PetscCall(MatShellSetOperation(j_shell, MATOP_MULT,
                                               (void (*)(void)) comp_step_deriv<true>));
                PetscCall(TSSetIJacobian(ts, j_shell, NULL, pre_jv_eval<true>, &ctx));
            }
            else
            {
                PetscCall(MatShellSetOperation(j_shell, MATOP_MULT,
                                               (void (*)(void)) comp_step_deriv<false>));
                PetscCall(TSSetIJacobian(ts, j_shell, NULL, pre_jv_eval<false>, &ctx));
            }
        }

        if (ctx.condition_system)
        {
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                                  "================================\n Condition system "
                                  "activated.\n================================\n"));
            PetscCall(TSSetTransientVariable(ts, primitive_to_conservative, &ctx));
            PetscCall(u_to_tilde(u_sol, u_sol));
            PetscCall(TSSetSolutionFunction(ts, petsc_sol_fun_tilde_var, &ctx));
        }
    }
    else
    {
        PetscCall(TSSetRHSFunction(ts, NULL, rhs_func, &ctx));
        if (ctx.complex_fd || ctx.condition_system)
        {
            PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: complex_fd and condition_system "
                                                    "options are ignored in explicit stepping.\n"));
        }
    }

    // Solve
    PetscCall(TSSetSolution(ts, u_sol));
    PetscCall(TSSetUp(ts));
    PetscCall(TSSolve(ts, u_sol));

    if (ctx.condition_system)
    {
        PetscCall(tilde_to_u(u_sol, u_sol));
    }

    PetscReal final_time{};
    PetscCall(TSGetTime(ts, &final_time));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Final Solution at t = %g:\n", (double) final_time));
    PetscCall(VecView(u_sol, PETSC_VIEWER_STDOUT_WORLD));

    // Cleanup
    if (j_shell) PetscCall(MatDestroy(&j_shell));
    PetscCall(VecDestroy(&ctx.u_curr));
    PetscCall(VecDestroy(&u_sol));
    PetscCall(TSDestroy(&ts));

    PetscCall(PetscFinalize());
    return 0;
}

/*TEST
  test:
    suffix: explicit
    args: -ts_monitor_error -ts_dt 0.1 -ts_type rk -ts_adapt_type none -pc_type none

  test:
    suffix: implicit
    args: -ts_monitor_error -ts_dt 0.01 -ts_type bdf -ts_bdf_order 2 -ts_adapt_type none
TEST*/
