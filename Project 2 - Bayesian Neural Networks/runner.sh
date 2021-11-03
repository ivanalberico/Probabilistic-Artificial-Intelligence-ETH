docker build --tag task2 .
docker run --rm -v "$( cd "$( dirname "$0" )" && pwd )":/code task2
