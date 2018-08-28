import os
import site
import sys
assert sys is not None

from esky import bdist_esky
assert bdist_esky is not None

from distutils.core import setup
from esky.bdist_esky import Executable

sys.argv.append("bdist_esky")

DESCRIPTION = 'Object detect '
AUTHOR = 'Tanveer'
AUTHOR_EMAIL = 'a@continental-corporation.com'
VERSION ="01.00.00"

BUILD_VERSION = "1.0"

FEATURES = [
    'New Feature: Mask detection ',
]

IS_BETA = False
app_name = 'mask_ecal_detect'

APPLICATION_NAME = '{0}.exe'.format(app_name)
if '--beta' in sys.argv:
    app_name = '{0}_Beta'.format(app_name)
    sys.argv.remove('--beta')
    IS_BETA = True

ADDITIONAL_FILES = [("category_index.py"), ("frozen_inference_graph_mask.py")]

INCLUDES = ['win32com', 'win32service', 'win32serviceutil',
            'win32event', 'psutil', 'simplejson'
            ]

# INCLUDES = []

EXCLUDES = ['Tkconstants', 'tcl', 'pdb', '_gtkagg',
            'bsddb', 'curses', 'pywin.debugger', '_gtkagg',
            '_tkagg', 'os2', 'doctest', 'pywin.debugger.dbgcon',
            'pywin.dialogs']

PACKAGES = ['cx_Oracle']

# default location
site_dir = site.getsitepackages()[1]
DYNAMIC_DLLS = []

if DYNAMIC_DLLS:
    tmp = []
    for dll in DYNAMIC_DLLS:
        tmp.append(os.path.join(site_dir, dll))
    ADDITIONAL_FILES.append(('.', tmp))

MAIN_SCRIPT = 'mask_ecal_detect.py'
# MAIN_SCRIPT = 'er.bat'

EXE = Executable(script=MAIN_SCRIPT,
                 gui_only=True,
                 name=app_name
                 )

print('=== Compiling {0} ==='.format(app_name))
setup(name=app_name,
      description=DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      version=BUILD_VERSION,
      data_files=ADDITIONAL_FILES,
      scripts=[EXE],
      options={'bdist_esky': {'excludes': EXCLUDES,
                              'includes': INCLUDES,
                              'bundle_msvcrt': True,
                              'freezer_module': 'cx_freeze',
                              'freezer_options': {
                                                  'packages': PACKAGES
                                                  },
                              },
               },
      )




