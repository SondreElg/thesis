from setuptools import setup

setup(
    name="pureples",
    version="0.0",
    author="Sondre Haugen Elgaaen",
    author_email="sondrehelgaaen@gmail.com",
    maintainer="Sondre Haugen Elgaaen",
    maintainer_email="sondrehelgaaen@gmail.com",
    url="https://github.com/SondreElg/thesis",
    license="MIT",
    description="Thesis project NTNU/UTokyo 2023",
    long_description="Thesis project for the Norwegian University of Science and Technology, written at the University of Tokyo in 2023"
    + "developed by Sondre Haugen Elgaaen."
    + "Based on Pureples library developed by Adrian Westh and Simon Krabbe Munck for evolving arbitrary neural networks. "
    + "HyperNEAT and ES-HyperNEAT is originally developed by Kenneth O. Stanley and Sebastian Risi",
    packages=[
        "pureples",
        "pureples/hyperneat",
        "pureples/es_hyperneat",
        "pureples/es_hyperneat_rnn",
        "pureples/es_hyperneat_ctrnn",
        "pureples/es_hyperneat_snn",
        "pureples/shared",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.x",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=["numpy", "neat-python", "graphviz", "matplotlib", "gym"],
)
