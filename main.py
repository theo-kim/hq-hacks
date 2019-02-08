#-- include('examples/showgrabbox.py')--#
import pyscreenshot as ImageGrab
from ctypes import windll
import pytesseract
import cv2
import numpy
from PIL import Image
import requests
from BeautifulSoup import BeautifulSoup
import webbrowser
from socketIO_client import SocketIO, LoggingNamespace
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import json
import wikipedia
import sys
import re

# For Development
import argparse

import nltk

class QueryNode() :
	def __init__(self, data=0, root=None, left=None, right=None) :
		self.parent = root
		self.left = left
		self.right = right
		self.data = data
	def __str__(self) :
		return "(" + str(self.data) + ")"
	def __repr__(self) :
		return "(" + str(self.data) + ")"
	def find(self, context) :
		return 1

class WordGroup(QueryNode) :
	def __init__(self, words=None, root=None) :
		if words is None :
			words = []
		QueryNode.__init__(self, words, root, None, None)
	def find(self, context) :
		output = []
		for i in self.data :
			output.append([])

		for i in range(len(self.data)) :
			word = self.data[i]
			for j in range(len(context)) :
				entry = context[j]
				if entry.count(word) > 0 :
					output[i].append(j)

		b = numpy.zeros([len(output),len(max(output,key = lambda x: len(x)))], dtype=int)
		b -= 20
		for i,j in enumerate(output):
			b[i][0:len(j)] = j
		
		self.matrix = b
		if len(b) > 1 : 
			self.score = numpy.sum(numpy.absolute(numpy.clip(numpy.absolute(numpy.subtract(b[0], b[1])), 1, 16) - 15))
		else: 
			self.score = 1
		return self.score
		# positions = numpy.sum(numpy.absolute(numpy.clip(numpy.absolute(numpy.subtract(b[0], b[1])), 1, 16) - 15))

class Operation(QueryNode) :
	def __init__(self, operation=None, root=None, left=None, right=None) :
		QueryNode.__init__(self, operation, root, left, right)
	def find(self, context) :
		if self.data == "AND" :
			self.score = (self.left.score + 1) * (self.right.score + 1)
			return self.score
		elif self.data == "->" :

			# print self.left
			# left_matrix = self.left.matrix
			# right_matrix = self.right.matrix

			# print numpy.sum(left_matrix, right_matrix)
			self.score = (self.left.score + self.right.score) / 2
			return self.score

def traverse(rootnode):
	thislevel = [rootnode]
	while thislevel:
		nextlevel = list()
		for n in thislevel:
			print n.data,
			if n.left: nextlevel.append(n.left)
			if n.right: nextlevel.append(n.right)
		print
		thislevel = nextlevel

def compileTree(stack) :
	curr = None
	j = 0
	while j < len(stack) :
		i = stack[j]
		ty = i.__class__.__name__
		if ty == "Operation" :
			i.left = curr
			curr.root = i
			curr = i
		elif ty == "WordGroup" :
			if curr is not None and curr.__class__.__name__ == "Operation" :
				curr.right = i
				i.root = curr
			else :
				curr = i
		elif i == "{" :
			outer = j
			for k in range(j + 1, len(stack)) :
				if stack[k] == "}" :
					outer = k
			temp = compileTree(stack[j + 1 : outer])
			j = outer
			if curr is not None and curr.__class__.__name__ == "Operation" :
				curr.right = temp
				temp.root = curr
			else :
				curr = temp
		j += 1
	return curr

def calc(tree, context):
    data = []

    def recurse(node) :
        if not node:
            return
        recurse(node.left)
        recurse(node.right)
        data.append(node.find(context))

    recurse(tree)
    return data[-1]

host = 'innes.xyz'
port = 80
# host = 'localhost'
# port = 3000

#with SocketIO(host, port, LoggingNamespace) as socketIO:
#    socketIO.emit('init')

user32 = windll.user32
user32.SetProcessDPIAware()

if __name__ == "__main__":

	"""
	START reading the screen
	"""
	# part of the screen
	# im=ImageGrab.grab(bbox=(10,550,1075,1500)) # X1,Y1,X2,Y2
	# open_cv_img = numpy.array(im.convert("RGB"))
	# open_cv_img = open_cv_img[:, :, ::-1].copy() 
	# gray = cv2.cvtColor(open_cv_img, cv2.COLOR_BGR2GRAY)

	# gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	# cv2.imwrite("screenshot.png", gray)

	# text = pytesseract.image_to_string(Image.open("screenshot.png"))

	# """
	# END reading the screen
	# """

	# """
	# START Parsing the question and the answer choices
	# """
	# split = text.split('\n')

	# question = ""

	# index = 0
	# top = len(split)
	# i = 0

	# while i < top :
	# 	if split[i] == "" or split[i] == " ":
	# 		split.pop(i)
	# 		i -= 1
	# 		top = len(split)
	# 	i += 1

	# while split[index].count('?') < 1 :
	# 	question += split[index] + ' '
	# 	index += 1;

	# print split

	# question += split[index] + ' '
	# index += 1
	# question_words = question.split(' ')
	# question_go_words = [word for word in question_words if word not in stopwords.words('english')]
	# question_stop_words = [word for word in question_words if word in stopwords.words('english')]
	
	# question_encoded = '+'.join(question.split(' '))
	# question_wiki = '_'.join(question.split(' '))
	# answer1 = split[index]
	# answer2 = split[index + 1]
	# answer3 = split[index + 2]

	# synonyms1 = [answer1, answer1.lower()]
	# synonyms2 = [answer2, answer2.lower()]
	# synonyms3 = [answer3, answer3.lower()]
	
	"""
	END Parsing the question and the answer choices
	"""
	# This is to analyze the answer choice synonyms
	# for syn in wn.synsets(answer1):
	#     for l in syn.lemmas():
	#     	if not l.name() in synonyms1 :
	# 	        synonyms1.append(l.name())
	# for syn in wn.synsets(answer2):
	#     for l in syn.lemmas():
	#     	if not l.name() in synonyms1 :
	# 	        synonyms2.append(l.name())
	# for syn in wn.synsets(answer3):
	#     for l in syn.lemmas():
	#     	if not l.name() in synonyms1 :
	# 	        synonyms3.append(l.name())

	"""
	START telling server that question has been parsed
	"""
	#with SocketIO(host, port, LoggingNamespace) as socketIO:
	#	socketIO.emit('info', { 'Q': question, 'A': answer1, 'B': answer2, 'C': answer3 })
	"""
	END telling server that question has been parsed
	"""

	"""
	START Determine question type 
	"""
	
	# 1. Determine word groups
	# 2. Determine word group meanings
	# 3. Determine word group logic
	# 4. Compile query
	
	#EX: Which fashion magazine is also the French word for "she" or "her"?
	# 1. [(Which), (fashion magazine), (is also) (the French word), (for "she" or "her")]
	# 2. 	(Which) -> choose from choices
	# 		(fashion magazine) -> subject #1
	#		(is also) -> alternative
	#		(the French word) -> descriptor #2
	#		(for "she" or "her") -> subject #2
	# 3.	
	# 4. (fashion magazine) AND (French word)->("she" "her") IN [ANSWER CHOICES]

	# Take query
	parser = argparse.ArgumentParser()
	parser.add_argument("query")
	args = parser.parse_args()
	query = args.query

	clause = query.split(' IN ');

	if len(clause) < 2 :
		print "You have to include an IN keyword to specify subject from operation"
		sys.exit()

	raw_subject = clause[1].replace('[', '').replace(']', '').split(', ')	
	procedure = clause[0]
	subject = []
	for i in raw_subject :
		subject.append(re.sub(r'\W+',' ', i))

	print "subject: " + str(subject)

	stack = []

	for i in range(len(procedure)) :
		s = procedure[i]
		if s == "(" :
			group = ""
			for j in range(i + 1, len(procedure)) :
				if procedure[j] == ")" :
					i = j
					break
				else :
					group += procedure[j]
			stack.append(WordGroup(group.split(', ')))
		elif s == "{" :
			stack.append("{")
		elif s == "}" :
			stack.append("}")
		elif procedure[i : i + 3] == "AND" :
			stack.append(Operation("AND"))
		elif procedure[i : i + 2] == "OR" :
			stack.append(Operation("OR"))
		elif procedure[i : i + 2] == "->" :
			stack.append(Operation("->"))
		
	procedure = compileTree(stack)
 
	index = 0

	hits = [0, 0, 0]

	while index < 3 :
		summ = None
		sel = subject[index]
		while summ is None :
			try:
				summ = wikipedia.page(sel)
				summ = summ.content.split(' ')
			except wikipedia.exceptions.DisambiguationError as e :
				options = str(e).split('\n')[1 :]
				j = []
				for i in options :
					j.append(calc(procedure, i.split(' ')))
				maximum = 0
				maximum_index = 0
				for i in range(len(j)) :
					if j[i] > maximum :
						maximum_index = i
						maximum = j[i]
				maximum_index
				sel = options[maximum_index]
			except :
				print "e"

		hits[index] = calc(procedure, summ)
		index += 1

	# print calc(procedure, summ)
	


	"""
	END Determine question type
	"""

	"""
	START calculating hit scores
	"""


	# r = requests.get("http://www.google.com/search?q=" + question_encoded + ' ' + answer1);
	# page = BeautifulSoup(r.text)
	# results = page.body.find('div', attrs={'id': 'resultStats'}).text
	# results = results.replace(',', '')
	# hits[0] = [int(s) for s in results.split(' ') if s.isdigit()][0]

	"""
	r = requests.get("http://www.google.com/search?q=" + question_encoded + ' ' + answer2);
	page = BeautifulSoup(r.text)
	results = page.body.find('div', attrs={'id': 'resultStats'}).text
	results = results.replace(',', '')
	hits[1] = [int(s) for s in results.split(' ') if s.isdigit()][0]

	r = requests.get("http://www.google.com/search?q=" + question_encoded + ' ' + answer3);
	page = BeautifulSoup(r.text)
	results = page.body.find('div', attrs={'id': 'resultStats'}).text
	results = results.replace(',', '')
	hits[2] = [int(s) for s in results.split(' ') if s.isdigit()][0]
	
	r = requests.get("http://www.google.com/search?q=" + question_encoded);
	page = BeautifulSoup(r.text)

	hit1 = 0
	hit2 = 0
	hit3 = 0

	for i in synonyms1 :
		hit1 += page.body.text.count(i)

	for i in synonyms2 :
		hit2 += page.body.text.count(i)

	for i in synonyms3 :
		hit3 += page.body.text.count(i)

	# hits[0] = hits[0] * ((hit1 * 1000) + 1);
	hits[0] = ((hit1 * 1000) + 1);
	# hits[1] = hits[1] * ((hit2 * 1000) + 1);
	hits[1] = ((hit2 * 1000) + 1);
	# hits[2] = hits[2] * ((hit3 * 1000) + 1);
	hits[2] = ((hit3 * 1000) + 1);
	
	# r = requests.get("https://en.wikipedia.org/w/api.php",
	# 	params={
	# 		'action': 'parse',
	# 		'format': 'json',
	# 		'page': answer3,
	# 		'prop': 'text',
	# 		'section': '0'
	# 	})
	# page = json.loads(r.text)['parse']['text']['*']

	# print page

	"""
	"""
	END calculating hit scores
	"""

	"""
	START determine and print the answer
	"""
	answer = ""

	# print answer1 + ": " + str(hits[0])
	# print answer2 + ": " + str(hits[1])
	# print answer3 + ": " + str(hits[2])

	if hits[0] > hits[1] and hits[0] > hits[2] :
		answer = "My Prediction: A"
		# requests.get("http://localhost:3000/set?answer=A")
		# requests.get("http://tk1931hqhacks.herokuapp.com/set?answer=A")
	elif hits[1] > hits[0] and hits[1] > hits[2] :
		answer = "My Prediction: B"
		# requests.get("http://localhost:3000/set?answer=B")
		# requests.get("http://tk1931hqhacks.herokuapp.com/set?answer=B")
	elif hits[2] > hits[1] and hits[2] > hits[0] :
		answer = "My Prediction: C"
		# requests.get("http://localhost:3000/set?answer=C")
		# requests.get("http://tk1931hqhacks.herokuapp.com/set?answer=C")
	elif hits[2] == hits[1] == hits[0] :
		answer = "I can't figure it out"
	elif hits[2] == hits[1] != hits[0] :
		answer = "Its either B or C"
	elif hits[2] != hits[1] == hits[0] :
		answer = "Its either A or B"
	elif hits[2] == hits[0] != hits[1] :
		answer = "Its either A or C"

	print answer;
	print hits
	# with SocketIO(host, port, LoggingNamespace) as socketIO:
 # 		socketIO.emit('answer', {'answer': answer})
 	"""
	END determine and print the answer
	"""
	# webbrowser.open("http://www.google.com/search?q=" + question_encoded);
	

#-#