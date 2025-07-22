#!/bin/bash  
  
# Source the setup script  
source /ryzers/setup.sh  
  
# Execute the command passed to the container  
exec "$@" 