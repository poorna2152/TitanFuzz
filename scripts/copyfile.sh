#!/bin/bash
set -euo pipefail

my_container="titanfuzz"
dest_dir="/home/src/run/"
echo "$dest_dir"

# Prepare destination directory inside container
docker exec -i -w /home/ "$my_container" rm -rf "$dest_dir" || true
docker exec -i -w /home/ "$my_container" mkdir -p "$dest_dir"

# Copy the entire workspace into the container dest directory
docker cp ./ "$my_container":"$dest_dir"

# Set permissions safely inside the container without overflowing argv and while handling spaces
# Directories: owner rwx, group/other rx; Files: owner rx, group/other rx
docker exec -i -w "$dest_dir" "$my_container" sh -c 'find . -type d -exec chmod 755 {} +'
docker exec -i -w "$dest_dir" "$my_container" sh -c 'find . -type f -exec chmod 555 {} +'

# Writable results directory
docker exec -i --user root -w "$dest_dir" "$my_container" mkdir -p Results
docker exec -i --user root -w "$dest_dir" "$my_container" chmod -R 777 Results

echo "Copy finished"
