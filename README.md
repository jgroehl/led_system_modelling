# LED System Modelling

This software can be used to model a photoacoustic imaging system, using LEDs as optical source.
The transducer elements are placed on a linear transducer array.

## Installation

### 1. Set up a Python environment

Set up a (virtual) Python environment for the project. To install all required Python packages,
run `pip install -r requirements`.

### 2. Install MCX
Install [MCX](https://mcx.space/). You can download suitable executables from their website,
but in order to access all custom sources that [SIMPA](https://github.com/IMSY-DFKZ/simpa) has implemented,
please build MCX yourself using [SIMPA's instructions](https://github.com/IMSY-DFKZ/simpa/tree/main#mcx-optical-forward-model).

### 3. Install MATLAB
Install [MATLAB](https://matlab.com/products/matlab.html), with the Image Processing Toolbox
and Parallel Computing Toolbox at minimum.

### 4. Install k-Wave
Download [k-Wave](http://k-wave.org/), as well as the C++/CUDA libraries. Detailed instructions kan be found on the
k-Wave website.

### 5. Install PACFISH
Clone the [PACFISH repository](https://github.com/IPASC/PACFISH). You might need to adapt `ipasc_simpa_kwave_adapter.py`
line 92, if your file hierarchy is different from the one shown below.

```
some-base-folder
 ├── led_system_modelling
 │   └── ...
 └── dependencies
     └── PACFISH
         └── ...
```

### 6. Adjust path configuration
The software will be looking for several paths in a `path_config.env` file. The PathManager will look for this file
in the following places (in this order):
1. The optional path given to the PathManager
2. Your home directory (`echo $HOME`)
3. The current working directory
4. The SIMPA home directory path

A default `path_config.env` is located in this repository, placing the file at location 3.

This path configuration will place simulation results in a `results` subfolder.
Additionally, it uses `matlab` and `mcx` as paths to the MATLAB and MCX executables, respectively.
You can adjust `path_config.env` if you wish to change any of this.

## Noise data
The software uses noise data as measured in a system without tissue to model noise.
You can place your noise file in `resources/NoiseMeasurement.mat`,
or change the relative path as defined in `run-simulation.py` (line 8).

## License
This software is licensed under [MIT License](LICENSE).
