# -*- mode: python -*-

block_cipher = None


a = Analysis(['lt5_tf_mask_detect.py'],
             pathex=['D:\\Work\\2018\\code\\Tensorflow_code\\object_detection\\tfl_masking_detect'],
             binaries=[],
             datas=[('frozen_inference_graph_mask.pb', '.'), ('category_index.json', '.'), ('_ecal_py_3_5_x64.pyd', '.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='lt5_tf_mask_detect.py',
          debug=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='lt5_tf_mask_detect.py')
