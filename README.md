# Keepsake: Taming the Chaos of Machine Learning Experiments

Ever trained a machine learning model and felt like you were wrestling a tangled mess of code, datasets, and hyperparameters? You're not alone.  Machine learning projects are exciting journeys of discovery, but keeping track of everything can feel like herding cats.  This is where Keepsake comes in as your friendly neighborhood version control system for machine learning.

Keepsake is an open-source Python library that acts like a magic wand for your ML experiments.  Imagine a tool that automatically tracks everything you throw at it – code, data, settings. No more scrambling to remember which version of the script you used, or what hyperparameters led to those mind-blowing results. Keepsake captures it all, making your life easier and your experiments more reproducible.

### Building a Movie Recommendation System with Keepsake

As an example, let's say you're building a movie recommendation system. You have a dataset filled with user IDs, movie IDs, and corresponding ratings.  Your mission:  to craft a model that predicts movie preferences for your users.

Here's a step-by-step walkthrough using Keepsake to keep your experiment organized and reproducible:

#### Installation:
Before diving in, you'll need to install Keepsake using pip, the Python package manager. Open a terminal or command prompt and run the following command:

``` bash
pip install -U keepsake
```

This will download and install the Keepsake library for you.

#### Train the Model with Keepsake Integration:
The core functionality of your experiment will remain within your Python script. Here's a breakdown of how Keepsake integrates with your training process:

1. Initialize Keepsake Experiment (experiment.init()):

This line marks the beginning of your Keepsake experiment. It tells Keepsake to start tracking everything related to this specific run of your model training script.

``` Python 
experiment = keepsake.init(
    path=".",  # Keepsake stores data in current directory by default
    params={"learning_rate": learning_rate, "num_epochs": num_epochs, "training_data": training_data_path},
)
```


* *path*: This argument specifies the location where Keepsake will store the experiment data. By default, it uses the current working directory.
* *params*: This dictionary stores the hyperparameters used for your model training, like learning rate and number of epochs. Keepsake will record these values for future reference.

2. Load Data and Train Model:

Load your movie rating dataset and train your chosen model (e.g., SVD in this case) as usual. Keepsake will silently track these details in the background.

```Python
dataset = Dataset.load_from_df(data[['userid', 'movieid', 'rating']], reader)
algo = SVD(lr_all=learning_rate, n_epochs=num_epochs)    
cv_results = cross_validate(algo, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

3. Checkpoint Experiment (experiment.checkpoint()):

Once the model is trained, Keepsake lets you create a checkpoint. This essentially saves a snapshot of the entire experiment, including the model file path and performance metrics:

```Python
experiment.checkpoint(
    path="svd_model.pkl",
    step=num_epochs,
    metrics={"RMSE": rmse_mean, "MAE": mae_mean},
    primary_metric=("RMSE", "minimize"),  # Track both RMSE and MAE
)
```

* *path*: This argument specifies the file or directory to be saved as part of the checkpoint. In this case, we're saving the trained model (svd_model.pkl).
* *step*: This argument allows you to create intermediate checkpoints during the training process. It's particularly useful for long-running training, especially in deep learning. By specifying a step value (e.g., an epoch number), Keepsake can save the model and metrics at that specific point in training.
* *metrics*: This dictionary stores the performance metrics calculated during training (e.g., RMSE, MAE). Keepsake will track these for future evaluation.
* *primary_metric*: This argument allows you to specify the primary metric to optimize. Keepsake can track multiple metrics, but this helps identify the most important one.

4. Stop Experiment (experiment.stop()):

This line signals the end of your Keepsake experiment. It's particularly important when working with Jupyter Notebooks, as Keepsake experiments running in scripts will eventually time out automatically. Calling experiment.stop() ensures proper cleanup and finalizes the experiment record.

#### Exploring Your Experiments: Keepsake CLI

Keepsake provides a powerful CLI that allows you to interact with your experiment data after training. Here's a quick glimpse of some essential commands:

`keepsake ls`: This command lists all the experiments stored within your current project directory.
<p align="center">
  <img width="696" alt="Command for showing lists of all experiments" src="https://github.com/dhwanisanmukhani/I3_blog/assets/28778091/0ded71ea-35b7-4fd6-b67d-49282d289395">
</p>


`(keepsake show <experiment_id>)`: This command displays detailed information about the experiment, including all the checkpoints under that experiment:

<p align="center">
<img width="555" alt="Command for showing all information about an experiment" src="https://github.com/dhwanisanmukhani/I3_blog/assets/28778091/676d3ea8-2f8e-4435-a9f9-766897bedb3a">
</p>

`keepsake rm <experiment_id>`: This command allows you to remove unwanted experiments from your Keepsake project. Use this command with caution, as deleted experiments cannot be recovered.

`keepsake checkout <experiment_id>`: If you need to access the code, model, or other files associated with a specific experiment, this command copies the contents of an experiment or checkpoint into your current project directory.

`keepsake diff <experiment_id>` This command provides a side-by-side comparison of two experiments, which helps you identify how changes in hyperparameters affected your model's performance.

<p align="center">
<img width="622" alt="Screenshot 2024-03-25 at 6 40 58 AM" src="https://github.com/dhwanisanmukhani/I3_blog/assets/28778091/9587852a-d33d-4a73-a694-bd2c10e8a0b9">
</p>

This glimpse only scratches the surface of Keepsake's capabilities. The CLI and python api offers a rich set of commands for advanced tasks, including but not limited to identifying the best experiment based on primary metrics, visualizing experiment comparisons with scatter plots, and much more. I encourage you to delve deeper into Keepsake's documentation to unlock its full potential and streamline your machine learning journey. https://keepsake.ai/docs/reference/python

### Strengths of Keepsake

Keepsake offers several advantages that streamline the machine learning experimentation process:
- Simplified Setup: Compared to other MLOps tools, Keepsake boasts a straightforward installation process using pip, the Python package manager. This eliminates complex configuration steps, allowing you to quickly integrate Keepsake into your existing workflow.
- Effortless Experiment Tracking: Keepsake seamlessly captures all aspects of your experiment, including code, data paths, hyperparameters, and even the training script itself. This eliminates the need for manual logging or meticulous record-keeping.
- Effortless Version Control: Similar to Git for code, Keepsake acts as a version control system for your ML experiments. You can create checkpoints throughout the training process, essentially saving snapshots of your experiment at different stages. This empowers you to effortlessly roll back to previous versions if needed, just like reverting to a specific commit in Git. This is particularly valuable for debugging, iterating on hyperparameters, or revisiting successful model configurations.
- Enhanced Reproducibility: By meticulously recording all experiment details, Keepsake ensures superior reproducibility. You can easily recreate past experiments with the exact same settings, facilitating comparisons and validation of results. This becomes crucial for sharing your work with others or replicating successful models in production environments.

### Limitations of Keepsake

While Keepsake offers significant benefits, it's essential to consider some potential limitations:

- Limited Integration: Currently, Keepsake primarily focuses on Python environments. Integration with other programming languages used in machine learning (e.g., R) is limited.
- Evolving Ecosystem: As a relatively new tool, Keepsake's ecosystem of integrations and community support is still under development. While it offers core functionalities, it might not yet have the extensive feature set of more established MLOps platforms.
- Limited Visualization: Keepsake currently offers limited built-in visualization capabilities for exploring experiment results. You might need to integrate external tools for in-depth analysis and data visualization.

In conclusion, Keepsake acts as your secret weapon for taming the chaos of machine learning experiments.  With its effortless setup, intuitive tracking, and seamless version control, Keepsake empowers you to experiment fearlessly, iterate effectively, and ensure the reproducibility of your findings. So, the next time you embark on an ML adventure, don't forget to pack Keepsake in your toolbox – it might just be the key to unlocking groundbreaking discoveries.
