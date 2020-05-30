#!/usr/bin/env python
# coding: utf-8

# In[5]:


codefile = open('/root/mlops_task_project/cnn_code.py','r')	#connecting to the code file
code = codefile.read()				                    #reading the code file

if 'keras' or 'tensorflow' in code:						#because keras or tensorflow keyword is a cmust have for a dl program
	if 'Dense' in code:				#beacuse if a code is of CNN ,then dense is must for adding layer
		print('CNN')
	else:
		print('not CNN')
else:
	print('not deep learning')


# In[ ]:




