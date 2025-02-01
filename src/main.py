#create in individual modules, import to here
import timedTest, untimedTest


print("RINGRING")
a = str()
while a.lower() not in ("untimed","timed"):
    a = input("What game?\n")
n = int(input("How many photos?\n"))
if a.lower() == "untimed":
    score, points, n = untimedTest.startGame(n)
else:
    score, points, n = timedTest.startGame(n)

print(f"You scored {score}/{n}, for {points} points!!")