from __future__ import generators
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def __str1__(self):
    desc = []
    if hasattr(self, 'integerSize'):
      desc.append('  Integer size: ' + str(self.integerSize//8) + ' bytes')
    if hasattr(self, 'dmswarmPIDSize'):
      desc.append('  DMSwarm pid size: ' + str(self.dmswarmPIDSize) + ' bytes (' + self.dmswarmPIDType + ')')
    return '\n'.join(desc)+'\n'

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-64-bit-indices=<bool>',   nargs.ArgBool(None, 0, 'Use 64-bit integers (long long) for indexing in vectors and matrices\n       This does not affect indexing for BLAS/LAPACK, see --with-64-bit-blas-indices'))
    help.addArgument('PETSc', '-with-dmswarm-pid-size=<int,long,long long>', nargs.Arg(None, 'long long', 'C integer type of the DMSwarm particle identifier field (DMSwarmField_pid)\n       Simulations with billions of particles need a 64-bit identifier (the default); int halves the per-particle memory of that field\n       This is independent of --with-64-bit-indices (int is always the C int type)'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers', self)
    self.libraries    = framework.require('config.libraries', None)
    self.compilers    = framework.require('config.compilers', None)
    self.types        = framework.require('config.types', self)
    return

  def fortranPromoteInteger(self):
      self.pushLanguage('FC')
      flags = self.getCompilerFlags()
      self.popLanguage()
      #   ifort compiler flag             gfortran compiler flag
      if '-integer-size 64' in flags or '-fdefault-integer-8' in flags:
        return 1
      return 0

  def configureIndexSize(self):
    '''Determine the size of PETSc indices (32 or 64-bit), from -with-64-bit-indices'''
    if self.framework.argDB['with-64-bit-indices']:
      self.integerSize = 64
      self.addDefine('USE_64BIT_INDICES', 1)
      if self.libraries.check('-lgcc_s.1', '__floatdidf'):
        self.compilers.LIBS += ' '+self.libraries.getLibArgument('-lgcc_s.1')
      self.addMakeMacro('PETSC_INDEX_SIZE', '64')
      if self.fortranPromoteInteger():
        self.addDefine('PROMOTE_FORTRAN_INTEGER', 1)
        self.logPrintWarning('You have a Fortran compiler option to promote integer to 8 bytes. This is fragile and not supported by the MPI standard. You must ensure in your code that all calls to MPI routines pass 4-byte integers.')
    else:
      self.integerSize = 32
      self.addMakeMacro('PETSC_INDEX_SIZE', '32')
      if self.fortranPromoteInteger():
        raise RuntimeError('Fortran compiler flag to promote integers to 8 bytes has been set, but PETSc is being built with 4-byte integers.')
    return

  def configureDMSwarmPIDSize(self):
    '''Determine the C integer type used for the DMSwarm particle identifier field, from -with-dmswarm-pid-size'''
    spellings = {'int': 'int', 'long': 'long', 'longlong': 'long long'}
    value     = str(self.framework.argDB['with-dmswarm-pid-size']).lower()
    for c in (' ', '\t', '-', '_'): value = value.replace(c, '')
    if value not in spellings:
      raise RuntimeError('--with-dmswarm-pid-size must be one of int, long, or "long long" (got "' + str(self.framework.argDB['with-dmswarm-pid-size']) + '")')
    self.dmswarmPIDType = spellings[value]
    self.dmswarmPIDSize = self.types.sizes[self.dmswarmPIDType.replace(' ', '-')]
    self.addDefine('USE_DMSWARM_PID_' + self.dmswarmPIDType.replace(' ', '_').upper(), 1)
    self.addDefine('DMSWARM_PID_SIZE', str(self.dmswarmPIDSize))
    self.logPrint('DMSwarm pid type is ' + self.dmswarmPIDType + ' (' + str(self.dmswarmPIDSize) + ' bytes)')
    return

  def configure(self):
    self.executeTest(self.configureIndexSize)
    self.executeTest(self.configureDMSwarmPIDSize)
    return
