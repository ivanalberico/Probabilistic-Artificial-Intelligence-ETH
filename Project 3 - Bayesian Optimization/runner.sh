docker build --tag task3 .
docker run --rm -v "$( cd "$( dirname "$0" )" && pwd )":/results task3