FROM cuda

RUN apt-get update
RUN apt-get install git -y
RUN apt-get install libboost-all-dev -y

# Make ssh dir
RUN mkdir /root/.ssh/
# Copy over private key, and set permissions
ADD id_rsa /root/.ssh/id_rsa
# Create known_hosts
RUN touch /root/.ssh/known_hosts
# Add bitbuckets key
RUN ssh-keyscan bitbucket.org >> /root/.ssh/known_hosts
# Clone repo
RUN git clone --recursive git@bitbucket.org:dzitkowskik/gpustore.git

WORKDIR /gpustore/install
RUN ./install_all.sh
RUN ldconfig

WORKDIR /gpustore/sample_data
RUN wget https://www.dropbox.com/s/3lea51f4jd2h2mz/openbookultraMM_N20130403_1_of_1
RUN wget https://www.dropbox.com/s/neej12spsmx2fhv/info.log

WORKDIR /gpustore
RUN cp config.ini.example config.ini
RUN make -j4 test
