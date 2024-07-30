# Use the TensorFlow image as the base
FROM tensorflow/tensorflow:2.15.0.post1-gpu-jupyter 

# Install Poetry
RUN pip install poetry

# Set up the working directory
WORKDIR /home/mrazo/git/scrappy

# Copy your project files
COPY pyproject.toml poetry.lock* /home/mrazo/git/scrappy/

# Install project dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of your project
COPY . /home/mrazo/git/scrappy

# Set the default command to open a bash shell
CMD ["/bin/bash"]