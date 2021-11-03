docker build --tag task4 .
docker run --rm -v "$( cd "$( dirname "$0" )" && pwd )":/results task4
