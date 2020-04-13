from distutils.core import setup

setup(name='Detection',
      version='0.1',
      description='Detect parts produced on 3D printer',
      author='Cézar Lemos',
      author_email='cezarcbl@protonmail.com',
      url='',
      packages=['detection'],
      install_recquires=['mxnet', 'gluoncv', 'opencv-python']
     )
