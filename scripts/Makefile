# Supplement voyage pilot: koopman_supplement_voyage_pilot.hpp
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

.PHONY: all clean run

all: koopman_supplement_mission

koopman_supplement_mission: koopman_supplement_mission.cpp koopman_supplement_voyage_pilot.hpp
	$(CXX) $(CXXFLAGS) -o $@ koopman_supplement_mission.cpp

run: koopman_supplement_mission
	./koopman_supplement_mission -o ../supplement_mission_log.csv

clean:
	rm -f koopman_supplement_mission
