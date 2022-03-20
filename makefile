MACRO=-DEBUG_DF
CXX=icpc
CXXFLAGS=  -fopenmp -O3 -std=c++14 

.PHONY :  kmtruss

kmtruss : 
	$(CXX) $(CXXFLAGS) -o kmtruss kmax_truss.cpp


.PHONY : clean
clean :
	rm kmtruss  *.o
