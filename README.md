# CSC_1004_Python_ImageClassification
Tutorial of Python Project

Student name: Yulun Wu
Student ID: 122090589
Github: https://github.com/albertkkk/CSC_1004_Python_ImageClassification
 
(Mention: This repository is different from the one I write in bi-weekly report, but I have already add TA into collaborator)


Brief introduction about my code and how do I implement the functions:

1.	training function:

	Teacher has already helped us to get each loss, so that the training accuracy is to add all the correct times together to divide total number.
For the training loss, I add all the loss together and divide the len(train_loader) 


2.	test function:
it is similar to the train function, without the backwards step. I just divide all the test correct examples by the number of test examples, getting the test accuracies.
Divide total loss by len(test_loader) to get test loss.

3.	plot function:
  
I identify the x, y as epochs and performance and call the plot function in the run function.
When each plot end, I add the plt.clf( ) to clear the previous images. And I save the filename by using the name of seed.

4.	write into txt:
 

In the run function, I let the code to write every number into relative txt file.

5.	Random seed and multiprocessing:
 
I put the multiprocessing in the code and one time run with three process which have different random seed. And I can get the result of three random seed at the same time.

6.	Plot mean function:
In order to get the mean. I read all the numbers from the txt file written before by using the readtxtname and clear functions which is added by myself.
   
Finally, in the plot mean function I call the readtxtname function to get the number into a new list. And add up the number in the new list and divide three to get the mean. And plot the mean image same as the plot function.
