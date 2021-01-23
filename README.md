# CS498 Spring 2021: AI for Robotic Manipulation

## Instructor: Kris Hauser

### Binder login

For the first few assignments, the most convenient way to get started is through Jupyter Notebook is through [this Binder environment](https://mybinder.org/v2/gh/krishauser/cs498ir_s2021/binder). Any time the class repository is changed, the first unlucky person to log on to Binder will encounter a lengthy build process that will take several minutes to complete. Subsequent logins will be much faster (about 20 seconds).

Once the environment launches, just navigate to the MPX/MPX.ipynb notebook in the appropriate folder to get started.

If you are working in Binder, *make sure to save your work to your local machine before closing the browser window*.  To do so, click on the "Save to browser storage" / "Restore from browser storage" buttons.

To download to your local machine for submission, click the "Download" button. To upload a notebook or code files, go back to the Jupyter file browser and choose the "Upload" button.

### Local installation

Local installation should be possible on Linux, Windows, and Mac.  Instructions are as follows:

1.  Install Git
2.  Install Python 3.x
3.  Install Python packages. On Linux / Mac this can be as easy as using pip

    ```
    python -m pip install klampt PyOpenGL numpy scipy scikit-learn matplotlib notebook
    ````

    For Windows, you may need to download the install files from this repository.
4.  Install the Klampt-jupyter-extension package.
    
    ```
    git clone https://github.com/krishauser/Klampt-jupyter-extension.git
    cd Klampt-jupyter-extension/jupyter-nbextension
    jupyter nbextension install klampt/ --user
    jupyter nbextension enable klampt/main 
    jupyter nbextension enable klampt/three.min 
    jupyter nbextension enable klampt/KlamptFrontend 
    cd ../..
    ```


You will be asked to periodically retrieve updated assignments; this will be done using

```
git pull
```

in your `cs498ir_s2021` directory.

To launch Jupyter, open a command line terminal, navigate to your `cs498ir_s2021` directory, and enter:

```
jupyter notebook
```

Once the environment launches, you may navigate to the MPX/MPX.ipynb notebook in the appropriate folder to get started.
