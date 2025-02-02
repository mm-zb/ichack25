#start game
#track score, points, n
#bonus points for geoguessr
#   run if fake && right
#show heatmap if fake

import random, os, csv

filepath = "src/static"

csvData = "train.csv"
#csv data


def startGame(n):
    images, answers = randomImAns(n)
    points = 0
    streak = 1
    score = 0
    for i in range(len(images)):
        ans, bonus = askQuestion(images[i],answers[i])
        if ans:
            points += 15*(streak*0.1+1)+bonus
            streak += 1
            score += 1
        else:
            streak = 1
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
    images = randomImages(int(n))
    answers = []
    for im in images:
        im = "train_data/"+im
        answers.append(ansMap[im])  
    return images, answers  



def randomImages(n):
    images = [f for f in os.listdir(filepath) if f.lower().endswith(('.jpg'))]
    return random.sample(images, n)


def askQuestion(real, image):
    bonus = 0
    if real=='1':
        heatmap = genHeatmap()
    show_image("images/"+image)
    print(image, real) #0 == real, 1 == fake
    print("Is this real or fake?")
    
    ans =  input()==str(real)
    if ans and real=='1':
        print("BONUS!")
        bonus = playGeoguessr(heatmap)

    return ans, bonus


def genHeatmap():
    return 0

def playGeoguessr(heatmap):
    return 10


def main():
    score, points, n = (startGame(int(input("How many questions?????"))))
    print(f"You scored {score}/{n}, for {points} points!!")