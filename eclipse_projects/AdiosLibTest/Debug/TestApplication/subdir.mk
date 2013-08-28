################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../TestApplication/CAdiosDataTestConstructor.cpp \
../TestApplication/MyClassTest.cpp \
../TestApplication/Test.cpp 

OBJS += \
./TestApplication/CAdiosDataTestConstructor.o \
./TestApplication/MyClassTest.o \
./TestApplication/Test.o 

CPP_DEPS += \
./TestApplication/CAdiosDataTestConstructor.d \
./TestApplication/MyClassTest.d \
./TestApplication/Test.d 


# Each subdirectory must supply rules for building sources it contributes
TestApplication/%.o: ../TestApplication/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"/Users/james.makela/Dev/GNOME/project_files/eclipse/AdiosLibTest/cute" -I"/Users/james.makela/Dev/GNOME/lib_adios" -I/Users/james.makela/Dev/boost_1_53_0 -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


