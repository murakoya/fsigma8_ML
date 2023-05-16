# Code for Machine Learning (CNN & NN)

These codes are used for the ML analysis in [arXiv number].

There are the CNN codes used for the image analysis of dark matter of jalo distribution in CNN,

nnd the NN codes used for the analysis of the legendre expanded power spectrum.

QUIJOTE simulation data are available from https://quijote-simulations.readthedocs.io/en/latest/access.html

## the contents in CNN (NN) directory

- cnn_model.py (nn_model.py)

    define CNN architecure. Please see Section 4.1 in our paper for details. (For NN, this file is named as nn_model.py)

- utils.py

  define some functions for loading data, making dataset for Pytorch,  and training and test our CNN.

- calc.py

    setting parameters and execution of training and test.

- results, and results/models directory

    the results of training and test saved in this directory.

- input_data directory

    a few samples are prepared in this directroy.


### parameters in calc.py for CNN

- input_dir

    you should prepare the 3D images in the directory specified by this parameter. The architecture in this directory should be

    ```
    ${input_dir}/
        ├ dir0
        |   ├ df.npy
        |   └ fsigma8.dat
        ├ dir1
        |   ├ df.npy
        |   └ fsigma8.dat
        ...
        └ dirN
            ├ df.npy
            └ fsigma8.dat
    ```

    , where X in the name 'dirX' means the label of realization, df.npy is the 3D image of dark matter or halo distribution, and the value of $f\sigma_8$ is written in fsigma8.dat.

- output_result

    the results of training and test are saved in the directory specified by this parameter.

- model_name

    traind CNN (NN) model is saved as ${model_name}.pkl in ${output_result} directory.

- test_file

    the outputs for the test images are written in the file \${output_result}/\${test_file}.

- gpu

    specify the number of gpu machine. This parameter depends on your machine.

- img_size

    the size of input images on a side in pixel. If you want to change this parameter from 40, we need to correct some numbers in `cnn_model.py`.

- batch_size

    specifies the batch size in training.

- ch

    the number of channel of input image. This value is 1 throughout our work.

- num_train, num_val

    Each parameter means the number of the training and validation data, respectively. The remaining data in ${input_dir} are used as the test data.

- num_epochs, lr, weight_decay
  Each means the number of training, the learning rate, and the parameter of the strangth of the penalty in L2 regularization.

- load_model
  When you re-train or test the CNN (NN), you can specify the model of your CNN (NN) used to retrain or test by this parametr.

- prediction

  When you set this parameter as True, training are skipped and the test data are input to the CNN (NN). And then, the outputs are written in \${output_result}/\${test_file}.

### parameters in calc.py for NN

Almost all of parameters are the same as the ones of CNN. The parameters different from CNN are explained.

- input_dir

    you should prepare the legendre axpanded power spectrum shared by QUIJOTE simulation project in the directory specified by this parameter. The architecture in this directory should be

    ```
    ${input_dir}/
        ├ dir0
        |   ├ Pk_m_RS0_z=0.5.txt
            ...
            ├ z05.dat
            ...
        |   └ fsigma8.dat
        ...
    ```

    , where X in the name 'dirX' means the label of realization, Pk_m_RS{Y}_z={Z}.txt is the Legendre expanded power spectrum (k, P0(k), P2(k), P4(k)) where Y is the specify the line of sight and Z is the value of redshift, and the value of $f\sigma_8$ is written in z{Z}.dat at z = {Z}.

- filename, labelfile

    you should specify the name of files which are written the power spectrum and the value of $f \sigma_8$.

- img_size

    This means the number of k-bins for power spectrum. the value '39' corresponds to $k < 0.25$ h/Mpc.




## How to use

After setting pf parameters, you can train or test the CNN or NN by doing

```
python calc.py
```
.

At the end of operation, the model after training and the model when the loss value is minimized in training are saved. And then, \${model_name}.pkl is copied into \${output_result}/models as modelYYYYMMDD_hour_minute.pkl.

This code's operation is checked for Python 3.6.9 and Pytorch 1.5.0.