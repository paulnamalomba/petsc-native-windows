from petsc4py import PETSc
from pathlib import Path
import tempfile
import unittest


class BaseTestPC:
    KSP_TYPE = None
    PC_TYPE = None
    COMM = PETSc.COMM_SELF

    def setUp(self):
        ksp = PETSc.KSP()
        ksp.create(self.COMM)
        pc = ksp.getPC()
        if self.KSP_TYPE:
            ksp.setType(self.KSP_TYPE)
        if self.PC_TYPE:
            pc.setType(self.PC_TYPE)
        self.ksp = ksp
        self.pc = pc

    def testAmatDef(self):
        pc_type = self.PC_TYPE
        for test_pc_type in ['mg']:
            pc = PETSc.PC().create()
            new_pc_type = 'none' if pc_type == test_pc_type else test_pc_type
            amat = []
            pc.setType(pc_type)
            amat.append(pc.getUseAmat())
            pc.setType(new_pc_type)
            amat.append(pc.getUseAmat())
            pc.setType(pc_type)
            amat.append(pc.getUseAmat())
            self.assertTrue(amat[0] == amat[2])
            self.assertFalse(amat[0] == amat[1])
            pc.destroy()

    def tearDown(self):
        self.ksp = None
        self.pc = None
        PETSc.garbage_cleanup()


class TestFIELDSPLITPC(BaseTestPC, unittest.TestCase):
    PC_TYPE = PETSc.PC.Type.FIELDSPLIT

    def testISoperations(self):
        test_index = [0, 1, 2]
        pc = self.pc
        is_u = PETSc.IS().createGeneral(test_index, comm=PETSc.COMM_SELF)
        pc.setFieldSplitIS(('u', is_u))

        self.assertTrue((pc.getFieldSplitSubIS('u').getIndices() == test_index).all())
        is_u = None


class TestMG(BaseTestPC, unittest.TestCase):
    PC_TYPE = PETSc.PC.Type.MG


class TestBDDCPC(BaseTestPC, unittest.TestCase):
    PC_TYPE = PETSc.PC.Type.BDDC

    def get_matis(self):
        local_mat = PETSc.Mat().createAIJ([3, 3], nnz=1, comm=PETSc.COMM_SELF)
        for i in range(3):
            local_mat.setValue(i, i, 2)
        local_mat.assemble()
        mat = PETSc.Mat().createIS([3, 3], comm=PETSc.COMM_SELF)
        mat.setISLocalMat(local_mat)
        local_mat.destroy()
        mat.assemble()
        return mat

    def testLoadSaveCustomization(self):
        mat = self.get_matis()
        boundary = PETSc.IS().createGeneral([0], comm=PETSc.COMM_SELF)
        self.pc.setOperators(mat)
        self.pc.setBDDCDirichletBoundariesLocal(boundary)
        boundary.destroy()
        self.pc.setUp()

        with tempfile.TemporaryDirectory() as tmpdir:
            for version in (PETSc.DECIDE, 0, 1):
                saved = Path(tmpdir) / f'bddc-customization-{version}.dat'
                resaved = Path(tmpdir) / f'bddc-customization-resaved-{version}.dat'
                if version == PETSc.DECIDE:
                    self.pc.saveBDDCCustomization(str(saved))
                else:
                    self.pc.saveBDDCCustomization(str(saved), version=version)

                loaded = PETSc.PC().create(PETSc.COMM_SELF)
                loaded.setType(PETSc.PC.Type.BDDC)
                loaded.setOperators(mat)
                if version == PETSc.DECIDE:
                    loaded.loadBDDCCustomization(str(saved))
                else:
                    loaded.loadBDDCCustomization(str(saved), version=version)
                loaded.setUp()
                if version == PETSc.DECIDE:
                    loaded.saveBDDCCustomization(str(resaved))
                else:
                    loaded.saveBDDCCustomization(str(resaved), version=version)

                self.assertEqual(saved.read_bytes(), resaved.read_bytes())
                loaded.destroy()
        mat.destroy()


class TestASMPC(BaseTestPC, unittest.TestCase):
    PC_TYPE = PETSc.PC.Type.ASM

    def checkLocalSubdomains(self, pc, indices, local_indices=None):
        # Set the subdomains on pc and check that they are returned unchanged.
        is_sub = [
            PETSc.IS().createGeneral(idx, comm=PETSc.COMM_SELF) for idx in indices
        ]
        is_local = None
        if local_indices is not None:
            is_local = [
                PETSc.IS().createGeneral(idx, comm=PETSc.COMM_SELF)
                for idx in local_indices
            ]
        pc.setASMLocalSubdomains(len(is_sub), is_sub, is_local)

        got_nsd, got_sub, got_local = pc.getASMLocalSubdomains()
        self.assertEqual(got_nsd, len(indices))
        self.assertEqual(len(got_sub), len(indices))
        for got, idx in zip(got_sub, indices):
            self.assertTrue((got.getIndices() == idx).all())

        if local_indices is None:
            self.assertEqual(len(got_local), 0)
        else:
            self.assertEqual(len(got_local), len(local_indices))
            for got, idx in zip(got_local, local_indices):
                self.assertTrue((got.getIndices() == idx).all())

        for iset in is_sub:
            iset.destroy()
        if is_local is not None:
            for iset in is_local:
                iset.destroy()

    def testLocalSubdomains(self):
        self.checkLocalSubdomains(self.pc, [[0, 1, 2], [3, 4, 5]])

    def testLocalSubdomainsWithLocalPart(self):
        self.checkLocalSubdomains(
            self.pc, [[0, 1, 2, 3], [2, 3, 4, 5]], [[0, 1], [4, 5]]
        )

    def testLocalSubdomainsParallel(self):
        # The index sets are in the global numbering of the vector, so give
        # each process a distinct block of it.
        rank = PETSc.COMM_WORLD.getRank()
        pc = PETSc.PC().create(PETSc.COMM_WORLD)
        pc.setType(PETSc.PC.Type.ASM)
        self.checkLocalSubdomains(pc, [[3 * rank, 3 * rank + 1, 3 * rank + 2]])
        pc.destroy()

    def testLocalSubdomainsCountOnly(self):
        # A subdomain count without index sets: nsd is set, the lists are not.
        self.pc.setASMLocalSubdomains(3)
        got_nsd, got_sub, got_local = self.pc.getASMLocalSubdomains()
        self.assertEqual(got_nsd, 3)
        self.assertEqual(got_sub, [])
        self.assertEqual(got_local, [])

    def testLocalSubdomainsUnset(self):
        # Nothing has been set and the preconditioner has not been set up, so
        # PETSc holds no subdomains yet.
        got_nsd, got_sub, got_local = self.pc.getASMLocalSubdomains()
        self.assertEqual(got_nsd, PETSc.DECIDE)
        self.assertEqual(got_sub, [])
        self.assertEqual(got_local, [])

    def testLocalScaling(self):
        # Independent weights on overlapping blocks reproduce the identity.
        A = PETSc.Mat().createAIJ([4, 4], nnz=1, comm=self.COMM)
        for i in range(4):
            A[i, i] = 1
        A.assemble()
        pc = self.pc
        pc.setOperators(A)
        self.checkLocalSubdomains(pc, [[2, 0, 1], [3, 2, 1]])
        pc.setASMType(PETSc.PC.ASMType.WEIGHTED)
        pc.setUp()
        nsd, is_sub, is_local = pc.getASMLocalSubdomains()
        self.assertEqual(nsd, 2)
        scaling = []
        values = ({0: 1, 1: 0.25, 2: 0.5}, {1: 0.75, 2: 0.5, 3: 1})
        for iset, weights in zip(is_sub, values):
            vec = PETSc.Vec().createSeq(iset.getLocalSize(), comm=self.COMM)
            vec.array[:] = [weights[i] for i in iset.getIndices()]
            scaling.append(vec)
        self.assertEqual(pc.getASMWeightedScaling(), [])
        pc.setASMWeightedScaling(scaling)
        got = pc.getASMWeightedScaling()
        self.assertEqual(len(got), nsd)
        for stored, vec in zip(got, scaling):
            self.assertTrue(stored.equal(vec))
        for stored in got:
            stored.destroy()
        # The setter retains the Vec objects independently of their wrappers.
        for vec in scaling:
            vec.destroy()
        x, y = A.createVecs()
        x.array[:] = [1, 2, 3, 4]
        pc.apply(x, y)
        self.assertTrue(y.equal(x))
        pc.applyTranspose(x, y)
        self.assertTrue(y.equal(x))
        x.destroy()
        y.destroy()
        for iset in is_sub + is_local:
            iset.destroy()
        A.destroy()


class TestHPDDMPC(BaseTestPC, unittest.TestCase):
    PC_TYPE = PETSc.PC.Type.HPDDM
    COMM = PETSc.COMM_WORLD

    @classmethod
    def setUpClass(cls):
        if not PETSc.Sys.hasExternalPackage('hpddm'):
            raise unittest.SkipTest('HPDDM unavailable, skipping tests')

    def testGeneo(self):
        args = ('n', 2)
        kargs = {'b': 3}
        track = {'cnt': 0}

        def assemble(J, t, X, X_t, s, ovl):
            track['cnt'] += 1

        def assemble_with_args(J, t, X, X_t, s, ovl, a1, a2, b=0):
            self.assertTrue(a1 == args[0])
            self.assertTrue(a2 == args[1])
            self.assertTrue(b == 3)
            track['cnt'] += 1

        # changing the order will fail the test, since we cannot
        # currently reset the callback to None
        assemble_func = [None, assemble, assemble_with_args]

        # Local and overlap size (ovl cannot be larger than nl)
        nl = 3
        novl = 1

        A = PETSc.Mat().create()
        A.setType('aij')
        A.setSizes(([nl, None],) * 2)
        A.assemble()
        self.pc.setOperators(A)

        rank = A.getComm().rank
        size = A.getComm().size
        naux = nl
        st = nl * rank
        if rank > 0:
            naux += novl
            st -= novl
        if rank < size - 1:
            naux += novl
        ovl = PETSc.IS().createStride(naux, st, comm=PETSc.COMM_SELF)
        Aux = PETSc.Mat().createAIJ(naux, comm=PETSc.COMM_SELF)
        Aux.assemble()
        for f in assemble_func:
            A.shift(1.0)
            Aux.shift(1.0)
            self.pc.setHPDDMHasNeumannMat(True)
            if f is assemble_with_args:
                self.pc.setHPDDMAuxiliaryMat(ovl, Aux, f, args=args, kargs=kargs)
            elif f is assemble:
                self.pc.setHPDDMAuxiliaryMat(ovl, Aux, f)
            else:
                self.pc.setHPDDMAuxiliaryMat(ovl, Aux)
            track = {'cnt': 0}
            self.pc.setUp()
            if f is not None:
                self.assertEqual(track['cnt'], 1)
            A.shift(1.0)
            self.pc.setUp()
            if f is not None:
                self.assertEqual(track['cnt'], 2)

        Aux.destroy()
        A.destroy()
        ovl.destroy()

    def testHPDDMSetters(self):
        pc = self.pc
        pc.setOptionsPrefix('hpddm_api_')
        opts = PETSc.Options(pc)
        for solver in ('eps', 'svd'):
            for dimension in ('ncv', 'mpd'):
                self.addCleanup(
                    opts.delValue, f'pc_hpddm_levels_1_{solver}_{dimension}'
                )

        pc.setHPDDMHarmonicOverlap(2)
        pc.setHPDDMEPSThreshold([0.1, 0.2])
        pc.setHPDDMEPSThreshold([0.5], relative=True)
        for option in (
            'pc_hpddm_harmonic_overlap',
            'pc_hpddm_levels_1_eps_threshold_absolute',
            'pc_hpddm_levels_1_eps_threshold_relative',
        ):
            self.assertFalse(opts.hasName(option))

        for solver, setter, count, ncv, mpd in (
            ('eps', pc.setHPDDMEPSDimensions, 'nev', 20, 15),
            ('svd', pc.setHPDDMSVDDimensions, 'nsv', 10, 8),
        ):
            prefix = f'pc_hpddm_levels_1_{solver}_'
            setter([5], [ncv], [mpd])
            self.assertFalse(opts.hasName(prefix + count))
            self.assertEqual(opts.getInt(prefix + 'ncv'), ncv)
            self.assertEqual(opts.getInt(prefix + 'mpd'), mpd)
            setter([PETSc.CURRENT], [PETSc.CURRENT], [PETSc.CURRENT])
            self.assertEqual(opts.getInt(prefix + 'ncv'), ncv)
            self.assertEqual(opts.getInt(prefix + 'mpd'), mpd)
            setter([PETSc.CURRENT], [PETSc.DETERMINE], [PETSc.DETERMINE])
            self.assertFalse(opts.hasName(prefix + 'ncv'))
            self.assertFalse(opts.hasName(prefix + 'mpd'))

    def testHPDDMSubKSP(self):
        A = PETSc.Mat().createAIJ(((3, None), (3, None)), nnz=1, comm=self.COMM)
        A.assemble()
        A.shift(1.0)
        pc = self.pc
        pc.setOperators(A)
        pc.setUp()
        ksp1 = pc.getHPDDMSubKSP(1)
        self.assertIsInstance(ksp1, PETSc.KSP)
        self.assertTrue(ksp1)
        ksp1.destroy()
        ksp1 = pc.getHPDDMSubKSP(1)
        self.assertTrue(ksp1)
        ksp1.destroy()
        x, y = A.createVecs()
        x.set(1.0)
        pc.apply(x, y)
        self.assertTrue(y.equal(x))
        x.destroy()
        y.destroy()
        A.destroy()

    def testHPDDMSolverModeSwitch(self):
        pc = self.pc
        pc.setOptionsPrefix('hpddm_mode_switch_')
        pc.setHPDDMHarmonicOverlap(1)
        A = PETSc.Mat().createAIJ(((3, None), (3, None)), nnz=1, comm=self.COMM)
        self.addCleanup(A.destroy)
        A.assemble()
        A.shift(1.0)
        pc.setOperators(A)
        x, y = A.createVecs()
        self.addCleanup(x.destroy)
        self.addCleanup(y.destroy)
        x.set(1.0)
        current = [PETSc.CURRENT]

        for relative in (False, True):
            with self.subTest(relative=relative):
                pc.setHPDDMEPSThreshold([0.5], relative=relative)
                pc.setHPDDMSVDDimensions([0], current, current)
                pc.setUp()
                ksp = pc.getHPDDMSubKSP(1)
                self.assertEqual(
                    ksp.getOptionsPrefix(), 'hpddm_mode_switch_pc_hpddm_coarse_'
                )
                ksp.destroy()
                pc.apply(x, y)
                self.assertTrue(y.equal(x))

        # SVD thresholds are parsed only when a coarse level is possible.
        if self.COMM.getSize() > 1:
            opts = PETSc.Options(pc)
            key = 'pc_hpddm_levels_1_svd_threshold_relative'
            opts[key] = 0.5
            try:
                pc.setFromOptions()
            finally:
                opts.delValue(key)
            pc.setHPDDMEPSDimensions([0], current, current)
            pc.setUp()
            ksp = pc.getHPDDMSubKSP(1)
            self.assertEqual(
                ksp.getOptionsPrefix(), 'hpddm_mode_switch_pc_hpddm_coarse_'
            )
            ksp.destroy()
            pc.apply(x, y)
            self.assertTrue(y.equal(x))

    def testHPDDMUnchangedHierarchy(self):
        pc = PETSc.PC().create(PETSc.COMM_SELF)
        pc.setType(PETSc.PC.Type.HPDDM)
        pc.setOptionsPrefix('hpddm_unchanged_')
        A = PETSc.Mat().createAIJ([3, 3], nnz=1, comm=PETSc.COMM_SELF)
        A.assemble()
        A.shift(1.0)
        pc.setOperators(A)
        pc.setHPDDMEPSThreshold([0.1])
        pc.setHPDDMEPSDimensions([1], [PETSc.CURRENT], [PETSc.CURRENT])
        pc.setFromOptions()
        # COMM_SELF limits the hierarchy to one level despite the requested nev.
        pc.setHPDDMEPSThreshold([PETSc.CURRENT])
        pc.setHPDDMEPSThreshold([0.1])
        pc.setHPDDMEPSDimensions([1], [PETSc.CURRENT], [PETSc.CURRENT])
        pc.setUp()
        ksp1 = pc.getHPDDMSubKSP(1)
        calls = (
            (pc.setHPDDMEPSThreshold, ([],)),
            (pc.setHPDDMEPSThreshold, ([PETSc.CURRENT],)),
            (pc.setHPDDMEPSThreshold, ([0.1],)),
            (pc.setHPDDMEPSDimensions, ([], [], [])),
            (pc.setHPDDMEPSDimensions, ([1], [PETSc.CURRENT], [PETSc.CURRENT])),
            (pc.setHPDDMEPSDimensions, ([PETSc.CURRENT],) * 3),
            (pc.setHPDDMSVDDimensions, ([], [], [])),
            (pc.setHPDDMSVDDimensions, ([PETSc.CURRENT],) * 3),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = Path(tmpdir) / 'hpddm.txt'
            for setter, args in calls:
                with self.subTest(setter=setter.__name__, args=args):
                    setter(*args)
                    pc.setUp()
                    same = pc.getHPDDMSubKSP(1)
                    self.assertEqual(same, ksp1)
                    same.destroy()
                    viewer = PETSc.Viewer().createASCII(
                        str(saved), comm=PETSc.COMM_SELF
                    )
                    pc.view(viewer)
                    viewer.destroy()
                    self.assertIn('level: 1\n', saved.read_text())
        ksp1.destroy()
        pc.destroy()
        A.destroy()

    def testHPDDMUnchangedDimensions(self):
        pc = self.pc
        pc.setOptionsPrefix('hpddm_unchanged_dimensions_')
        pc.setHPDDMHarmonicOverlap(1)
        opts = PETSc.Options(pc)
        A = PETSc.Mat().createAIJ(((3, None), (3, None)), nnz=1, comm=self.COMM)
        self.addCleanup(A.destroy)
        A.assemble()
        A.shift(1.0)
        pc.setOperators(A)
        for solver, setter in (
            ('eps', pc.setHPDDMEPSDimensions),
            ('svd', pc.setHPDDMSVDDimensions),
        ):
            for dimension in ('ncv', 'mpd'):
                self.addCleanup(
                    opts.delValue, f'pc_hpddm_levels_1_{solver}_{dimension}'
                )
            for ncv, mpd in ((3, 2), (PETSc.DETERMINE, PETSc.DETERMINE)):
                setter([0], [ncv], [mpd])
                pc.setUp()
                ksp = pc.getHPDDMSubKSP(1)
                before = ksp.getId()
                ksp.destroy()
                for args in (
                    ([PETSc.CURRENT], [ncv], [PETSc.CURRENT]),
                    ([PETSc.CURRENT], [PETSc.CURRENT], [mpd]),
                    ([0], [ncv], [mpd]),
                ):
                    with self.subTest(solver=solver, args=args):
                        setter(*args)
                        pc.setUp()
                        ksp = pc.getHPDDMSubKSP(1)
                        after = ksp.getId()
                        ksp.destroy()
                        self.assertEqual(before, after)

    def testHPDDMSingleRankSetters(self):
        A = PETSc.Mat().createAIJ([3, 3], nnz=1, comm=PETSc.COMM_SELF)
        self.addCleanup(A.destroy)
        A.assemble()
        A.shift(1.0)
        x, y = A.createVecs()
        self.addCleanup(x.destroy)
        self.addCleanup(y.destroy)
        x.set(1.0)
        current = [PETSc.CURRENT]
        for fromoptions in (False, True):
            for solver in ('eps', 'svd', 'threshold'):
                with self.subTest(fromoptions=fromoptions, solver=solver):
                    pc = PETSc.PC().create(PETSc.COMM_SELF)
                    self.addCleanup(pc.destroy)
                    pc.setType(PETSc.PC.Type.HPDDM)
                    pc.setOptionsPrefix('hpddm_single_rank_')
                    pc.setOperators(A)
                    if fromoptions:
                        pc.setFromOptions()
                    if solver == 'svd':
                        pc.setHPDDMHarmonicOverlap(1)
                    for count in (1, 2):
                        if solver == 'eps':
                            pc.setHPDDMEPSDimensions([count], current, current)
                        elif solver == 'svd':
                            pc.setHPDDMSVDDimensions([count], current, current)
                        else:
                            pc.setHPDDMEPSThreshold([0.1 * count])
                        pc.setUp()
                        ksp = pc.getHPDDMSubKSP(1)
                        self.assertEqual(
                            ksp.getOptionsPrefix(),
                            'hpddm_single_rank_pc_hpddm_coarse_',
                        )
                        ksp.destroy()
                        pc.apply(x, y)
                        self.assertTrue(y.equal(x))


if __name__ == '__main__':
    unittest.main()
