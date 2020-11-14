var1=1000
for N in $(seq -w 10000 69999); do
  mv "$N.jpg" "$((N/var1))000"
done
