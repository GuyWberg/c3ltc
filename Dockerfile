FROM sagemath/sagemath:latest

WORKDIR ./home

ADD . .

RUN sudo apt-get update

RUN sudo apt-get install -y build-essential autoconf libtool autotools-dev git

RUN sh setup_spasm.sh

RUN sage -pip install -r requirements.txt

CMD [ "sage", "-n", "jupyter", "--ip", "0.0.0.0", "--no-browser" ]
