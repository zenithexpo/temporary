---
layout: post
title: "Setting Up a Private Docker Registry : A Complete guide"
subtitle: A private Docker registry gives you better performances for big clusters and high-frequency roll-outs, plus added features like access authentication.
tags: [blog, Docker, Docker-Registry, Tech-Blog]
image: /blog/blog_img/docker_reg/docker.png
share-img: "/blog/blog_img/docker_reg/reg.png"
permalink: /2020/04/22/SettingUpPrivateDockerRegistry/
gh-repo: imSrbh/Docker-Registry
gh-badge: [fork, star, watch, follow]
---
Docker has it's own free docker registry [Docker Hub](https://hub.docker.com/) from where anyone can push or pull the images.  

![](/blog/blog_img/docker_reg/reg.png)

**But Why Private docker registry comes into picture...?**  
A private Docker registry allows you to share your custom base images within your organization, keeping a consistent, private, and centralized source of truth for the building blocks of your architecture. A private Docker registry gives you better performances for big clusters and high-frequency roll-outs, plus added features like access authentication.

### Tasks -

- [x] Setup Local Docker Registry using Docker.
- [x] Add Basic Authentication to this Docker Registry.
  - [x] docker-compose file.
  - [ ] We can write a script that fetch the username and password from DB or import all your users using htpasswd.
- [ ] To have Role Based Multi User Authentication for Local Docker Registry get this up and running with docker-compose.



## 1. a) Setup a Local Docker Registry (insecure)

- Pull the registry [image](https://hub.docker.com/_/registry) from docker hub by running  
`docker pull registry`

- Run a container of this image  
  `docker container run -d -p 5000:5000 --name simple_registry  registry`

- Check in the browser  
[http://127.0.0.1:5000/v2/_catalog](http://127.0.0.1:5000/v2/_catalog)  
or  
[http://localhost:5000/v2/_catalog](http://localhost:5000/v2/_catalog)

**Note :**

- Now you can inspect the  Container

   `docker container inspect <container>`

  You will see volume attached with this container here `/var/lib/registry`

  ```json
  .
  .
  .
  "Mounts": [
              {
                  "Type": "volume",
                  "Name": "5ae483d077d9113b246b08e2a6d8b2afe800a23912c234d52f5b0fed202101e9",
                  "Source": "/var/lib/docker/volumes/5ae483d077d9113b246b08e2a6d8b2afe800a23912c234d52f5b0fed202101e9/_data",
                  "Destination": "/var/lib/registry",
                  "Driver": "local",
                  "Mode": "",
                  "RW": true,
                  "Propagation": ""
              }
          ],
  .
  .
  .
  ```

  if somehow this container stops,  then spin up a new container of the image and give this mount point `/var/lib/registry` to that container. so that you can retrieve the data/images of the container.

- Push some images to your local docker registry  
```
server@ubuntu:~$ docker images
REPOSITORY                         TAG                 IMAGE ID            CREATED             SIZE
ubuntu                             latest              1d622ef86b13        2 weeks ago         73.9MB
```

  1. give a new tag name the image with `localhost:5000/` prefix so that it will be pushed to your docker registry.

     ` sudo docker image tag ubuntu:latest localhost:5000/ubuntu`

  2. Push this image to the local docker registry.

     `sudo docker push localhost:5000/ubuntu`

- check the browser refresh.

  [http://localhost:5000/v2/_catalog](http://localhost:5000/v2/_catalog)


**Note :**  

- By default docker will only allow secure registry, But this is exception for this subnet 127.0.0.0/8 CIDR. You can't push or pull if your IP range differ from this range.

  for my system's IP _ifconfig_ 192.168.123.136 (this is insecure)

  for this IP we need to add this then only we can push or pull...

  1. cd /etc/docker/

  2. Create a file `daemon.json`

  3. Give the server IP

  ```json
  {
  	"insecure-registries" : ["192.168.123.136:5000"]
  }
  ```

---

## b) Setup a Secure docker registry (certificate based)

- Create certificates and store the certificates.

- For that create a directory certs.

  ```sh
  mkdir certs
  ```

  ```sh
  openssl req -newkey rsa:4096 -nodes -sha256 -keyout certs/domain.key -x509 -days 365 -out certs/domain.crt
  ```



  

- After hitting above cert gen code in in command line you need to enter some details.
  so press enter for all the fields but in Common Name :
  `Common Name (e.g. server FQDN or YOUR name) []:repo.docker.local`

  anyone else who will be accessing the registry he should access it at `repo.docker.local `  url otherwise it will not be accessible.
  
  >We can use any name instead of local `repo.docker.xyz`. the only thing we have to take care is /etc/hosts we should add `<ip_addr> repo.docker.xyz`  

- And now follow these steps...
  ```sh
  cd /etc/docker/
  mkdir certs.d
  
  cd certs.d/
  mkdir repo.docker.local:5000
  
  Copy the domain.crt file.
  cp certs/domain.crt /etc/docker/certs.d/repo.docker.local\:5000/ca.crt
  ```

- **restart docker service...**

  `service docker restart`

- **Start the container...**

  ```shell
  docker container run -d -p 5000:5000 --name secure_registry -v $(pwd)/certs/:/certs -e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/domain.crt -e REGISTRY_HTTP_TLS_KEY=/certs/domain.key registry
  ```

- Rename tag to push the image.

  `sudo docker tag ubuntu repo.docker.local:5000/ubuntu`

- Push the image.

  `docker image push repo.docker.local:5000/ubuntu`

  **you will see...**

  ```sh
  Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post http://%2Fvar%2Frun%2Fdocker.sock/v1.40/images/repo.docker.local:5000/ubuntu/push?tag=: dial unix /var/run/docker.sock: connect: permission denied
  ```

  because this doesn't resolve this name repo.docker.local

  so for this we need to add <ip_addr> repo.docker.local in `/etc/hosts`

  add this`192.168.123.136 repo.docker.local`

again run the above docker push command. And this time it will be successfully pushed.

**So this was all about to set up secure docker registry.**

---

## 2. Setting up Docker Registry With Basic Authentication

- Create a auth directory to store the htpasswd

`mkdir auth`

docker container run --entrypoint htpasswd registry -bnB <username> <password> >auth/htpasswd

_eg._

`docker container run --entrypoint htpasswd registry -bnB saurabh password >auth/htpasswd`



-bnB for

**b** - run in batch mode  
**n** - output should be displayed  
**B** - bcrypt (passwd will be encrypted in this fashion)  

you can see this htpasswd file which stored the username and bcrypted password. 

```
  server@ubuntu:~$ cat /auth/htpasswd
  saurabh:$2y$05$VzJ.ud8r06fYO/V/SV7dfOpb2i5ipU5E05IdtaDbW8Py15e.y8ixO
  ```
* Spin up a container 

```sh
  docker container run -d \
  -p 5000:5000 \
  --name registry \
  -v "$(pwd)"/auth:/auth \
  -v "$(pwd)"/certs:/certs \
  -e "REGISTRY_AUTH=htpasswd" \
  -e "REGISTRY_AUTH_HTPASSWD_REALM=Registry Realm" \
  -e REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd \
  -e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/domain.crt \
  -e REGISTRY_HTTP_TLS_KEY=/certs/domain.key \
  registry
```

  


Now your container is Up and Running.

so you will see this error when you try to push image after running the registry container...
```
server@ubuntu:~$ sudo docker image push repo.docker.local:5000/ubuntu
The push refers to repository [repo.docker.local:5000/ubuntu]
16542a8fc3be: Preparing 
6597da2e2e52: Preparing 
977183d4e999: Preparing 
c8be1b8f4d60: Preparing 
no basic auth credentials
```
So now you need to login...

`docker login repo.docker.local:5000`

it will ask Username and password

```
server@ubuntu:~$ sudo docker login repo.docker.local:5000
Username: saurabh
Password: 
Error saving credentials: error storing credentials - err: exit status 1, out: `Error calling StartServiceByName for org.freedesktop.secrets: GDBus.Error:org.freedesktop.DBus.Error.TimedOut: Failed to activate service 'org.freedesktop.secrets': timed out (service_start_timeout=120000ms)`
```



I faced the above error while login because it was not able to store credintials...

`sudo apt install gnupg2 pass`

This worked for me..

so now you can push the image.

`sudo docker image push repo.docker.local:5000/ubuntu`

---

### Deploy the registry with basic authentication using a Compose file

`docker-compose.yml`

```yaml
registry:
  restart: always
  image: registry:2
  ports:
    - 5000:5000
  environment:
    REGISTRY_HTTP_TLS_CERTIFICATE: /certs/domain.crt
    REGISTRY_HTTP_TLS_KEY: /certs/domain.key
    REGISTRY_AUTH: htpasswd
    REGISTRY_AUTH_HTPASSWD_PATH: /auth/htpasswd
    REGISTRY_AUTH_HTPASSWD_REALM: Registry Realm
  volumes:
    - /path/data:/var/lib/registry
    - /path/certs:/certs
    - /path/auth:/auth
```

- Replace `/path` with the directory which contains the `certs/` and `auth/` directories.

- Start your registry by issuing the following command in the directory containing the `docker-compose.yml` file:

  ```
  $ docker-compose up -d
  ```

___

 **References :**  
* [Docker Registry](https://docs.docker.com/registry/)
* [Digital Ocean](https://www.digitalocean.com/community/tutorials/how-to-set-up-a-private-docker-registry-on-ubuntu-18-04)
