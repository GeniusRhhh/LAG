integer_num=42
float_num=3.14
string="Hello"
boolean=True
none_value=None

my_list=[1,2,3,"mixed"]
my_tuple=(4,5,6)
my_dict={"name":"Alice","age":25}
my_set={1,2,3,3}

x=10
if x>0:
    print("x is positive")
elif x==0:
    print("x is zero")
else:
    print("x is negative")


for item in my_list:
    print(f"Item:{item}")

c=0
while c<3:
    print(f"c:{c}")
    c+=1

def calculate(a,b=0):
    return a+b

result=calculate(5,3)
print(f"result:{result}")

squares=[x**2 for x in range(5)]
print(f"squares:{squares}")

class Person:
    def __init__(self,name,age):
        self.name=name
        self.age=age

    def introduce(self):
        return f"Hi,I'm {self.name},{self.age} years old"

alice=Person("Alice",25)
print(alice.introduce())

try:
    result=10/0
except ZeroDivisionError as e:
    print(f"Error:{e}")
finally:
    print("This always runs")

double = lambda x:x*2
print(f"Double {double(5)}")

def add(a,b):
    return a+b

print(add(1,2))
print(add("Hello","World"))

def kongCaoZuo():
    pass