import os
def cclda(file,iterations,topics,outputfile):
   os.system('javac *.java')
   os.system('java LearnTopicModel -model cclda -input {} -iters {} -Z {}'.format(file,iterations,topics))
   output = file + '.assign'
   os.system('python topwords_cclda.py {} > {}'.format(output,outputfile))

os.system('rm *.class')
cclda('input.txt',10,10,'output.txt')
