#start at 10 seconds per question
#take in images as list of n length
#second list of real or not (1 or 0)
#display image
#ask question function
#   if timer ends, return False
#   return true if question correct
#
import random, os, threading, csv
from PIL import Image

filepath =  "images/"

csvData = "train.csv"
#csv data


def startGame(n):
    images, answers = randomImAns(n)
    tpq = 6 #time per question
    points = 0
    streak = 1
    score = 0
    for i in range(len(images)):
        ans = askQuestion(images[i],answers[i], tpq)
        if ans:
            points += 15*(streak*0.1+1)
            streak += 1
            if tpq >= 2:
                tpq -= int(points*0.05)
            score += 1
        else:
            streak = 1
            if tpq <= 10:
                tpq += int(points*0.05)
    return score, points, n

def answerDict():
    answers = {}
    with open(csvData, "r", newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            answers[row['file_name']] = row['label'] #makes a dict of filename to answer
    return answers
    
def randomImAns(n):
    ansMap = answerDict()
    answers = []
    images = randomImages(n)
    for im in images:
        f = "train_data/"+im
       # print(list(ansMap.keys())[:3])
        #print(f)
       # print(list(ansMap.values())[:3])
        #print(ansMap)
        answers.append(ansMap[f])
    return answers, images


def randomImages(n):
    images = [f for f in os.listdir(filepath) if f.lower().endswith(('.jpg'))]
    return random.sample(images, n)

def show_image(filepath):
    img = Image.open(filepath) 
    img.show() 

def askQuestion(real, image, timeout):
    answer = [None]
    stop_event = threading.Event()

    def get_input():
        show_image("images/"+image)
        print(image, real) #0 == real, 1 == fake
        print(timeout, "seconds!")
        print("Is this real or fake?")
        if not stop_event.is_set():
            answer[0] = input()
    
    thread = threading.Thread(target=get_input, daemon=True)
    thread.start()
    thread.join(timeout)  

    if thread.is_alive():
        stop_event.set()
        print("\nTime's up!")
        return False  # return False if the user didn't answer in time
    
    return (answer[0]==str(real))

def main():
    score, points, n = (startGame(int(input("How many questions?????"))))
    print(f"You scored {score}/{n}, for {points} points!!")