# Get all files in current directory having names XXXXsomename, where X is an integer
files=$(find . -name 'antrep[0-9][0-9]seed1*')

# Build a list of the XXXX patterns found in the list of files
for name in ${files}; do
  mv name "/nfs/kun2/users/hanqi2019/1203complete/ant/seed1"
done

# Return from script with normal status
exit 0