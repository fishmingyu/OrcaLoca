BUILD_ARGS_COMMON="
	--build-arg PLATFORM --rm -t orcar/${PLATFORM}:${TAG} -f Dockerfile .
"

docker build ${BUILD_ARGS_COMMON}