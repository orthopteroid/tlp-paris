# linear program solver (simplex method), orthopteroid@gmail.com, BSD2 license
# based on "Operations Research", Hiller & Lieberman (1974)

cmake_minimum_required(VERSION 3.13)
project(tinylp C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(tinylp
        tinylp.c tinylp.h
        analysis.c analysis.h
        main.c
        )
