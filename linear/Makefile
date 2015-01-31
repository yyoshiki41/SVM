#
#     リンク
#
CFLAGS = -c -Wall -O2
OBJECT = data.o main.o
pgm: $(OBJECT)
	clang++ -std=c++11 $(OBJECT) -o main.out -lm
#
#     コンパイル
#
main.o: main.cpp
	clang++ -std=c++11 $(CFLAGS) main.cpp
data.o: data.cpp
	clang++ -std=c++11 $(CFLAGS) data.cpp
