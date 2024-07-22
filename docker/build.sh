BUILD_ARGS_COMMON="
	--build-arg PLATFORM --build-arg OPENAI_API_KEY --rm -t orcar/${PLATFORM}:${TAG} -f Dockerfile .
"

docker build ${BUILD_ARGS_COMMON}