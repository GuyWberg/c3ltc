git clone https://github.com/cbouilla/spasm;
cp kernel.c ./spasm/test;
cd spasm;
autoreconf -i;
./configure && make;
cd test;
make kernel;