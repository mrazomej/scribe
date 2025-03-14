# Use the NVIDIA JAX image as the base
FROM nvcr.io/nvidia/jax:25.01-py3

# Install zsh and other required packages
RUN apt-get update && apt-get install -y \
    zsh \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Install your favorite Oh My Zsh plugins
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# set zsh theme
RUN sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="avit"/' ~/.zshrc

# Set up the working directory
WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH=/root/.local/bin:$PATH

# Create a virtual environment instead of using system Python
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/venv

# Add the virtual environment to PATH
ENV PATH=/app/venv/bin:$PATH

# Copy dependency files first
COPY pyproject.toml ./
COPY uv.lock* ./

# Install dependencies into the virtual environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install numpyro && \
    uv pip install -e ".[dev]"

# Copy the rest of your project
COPY . .

# Set the user and group ID (these will be filled in when building)
ARG USER_ID
ARG GROUP_ID

# Set the locale for UTF-8 used for Sphinx docs
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Ensure all files are owned by the user
RUN chown -R ${USER_ID}:${GROUP_ID} /app /app/venv

USER ${USER_ID}:${GROUP_ID}

# Set the default command
SHELL ["/bin/zsh", "-c"]
ENTRYPOINT ["zsh"]