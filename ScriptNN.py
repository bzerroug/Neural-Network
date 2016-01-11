import DataProcessing as dp
import network as NN

train = dp.get_data_train("mnist_train.csv")
test = dp.get_data_test("mnist_test.csv")
print "*************************"
print "* train and test loaded *"
print "*************************"
nn = NN.Network([784,30,10])
print "*************************"
print "*      nn loaded        *"
print "*************************"
nn.SGD(train,30,10,0.1,test_data=test)
