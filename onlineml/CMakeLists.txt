
set(
	HEADER_FIELS
	classifier.h
	classifiers/perceptron.h
	data.h
	onlineml.h
	parameter.h
	string_process.h
	trainer.h
)

set(
	SOURCES
	classifier.cc
	classifiers/perceptron.cc
	data.cc
	parameter.cc
	trainer.cc
)

add_library(onlineml SHARED ${SOURCES} ${HEADER_FIELS})
target_include_directories(onlineml PUBLIC ${HEADER_FIELS})

add_subdirectory(onlineml_cli)
