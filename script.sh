#!/bin/bash

# Check if a filename is provided as argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

# Check if the file exists
if [ ! -f "$1" ]; then
    echo "File $1 does not exist"
    exit 1
fi

# Get file size
file_size=$(stat -c %s "$1")

# Get file content and remove newlines
file_content=$(tr -d '\n' < "$1")

# Escape double quotes in file content
file_content_escaped=$(echo "$file_content" | sed 's/"/\\"/g')

# Construct JSON string
json_string="{\"file_size\": $file_size, \"file_content\": \"$file_content_escaped\"}"

# Output JSON string
echo "$json_string" > message.json
