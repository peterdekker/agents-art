# agents-art
## Installation
This software requires running Python 3.7 in a virtualenv. To this end, we use the tool pyenv:

```
# Install build dependencies (specific for Ubuntu)
sudo apt-get update; sudo apt-get install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Download pyenv
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cd ~/.pyenv && src/configure && make -C src

# Set environment variables
# the sed invocation inserts the lines at the start of the file
# after any initial comment lines
sed -Ei -e '/^([^#]|$)/ {a \
export PYENV_ROOT="$HOME/.pyenv"
a \
export PATH="$PYENV_ROOT/bin:$PATH"
a \
' -e ':a' -e '$!{n;ba};}' ~/.profile
echo 'eval "$(pyenv init --path)"' >>~/.profile

echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Restart shell (or is full relogin needed?)
exec $SHELL

# install Python 3.7.12
pyenv install 3.7.12

# Install pyenv-virtualenv
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
exec "$SHELL"

# Create virtualenv with Python 3.7
pyenv virtualenv 3.7.12 env37
#Activate environment
pyenv activate env37
# Install requirements
pip3 install -r requirements.txt
# Create jupyter notebook kernel for this env
python3 -m ipykernel install --user --name env37 --display-name "env37"
```