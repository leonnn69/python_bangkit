#!/bin/sh
# loops on bash
n = 1 
while [ $n -le5 ];do
    echo "Iration number $n"
    ((n+1))
done

# conditional on bash
ig grep "127.0.0.1" /etc/hosts; then
        echo "everything okey"
else
        echo "error! 127.0.0.1 not in /etc/hosts"
fi

# for in bash
for fruit in peach banana apple; do
        echo "i like $fruit"
done
