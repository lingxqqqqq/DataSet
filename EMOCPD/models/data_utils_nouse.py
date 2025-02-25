from __future__ import division
from __future__ import print_function
import tables
import numpy
import sys
import tables
from tables import open_file

DEFAULT_NODE_NAME = "defaultNode"


def init_h5_file(toDiskName, groupName=DEFAULT_NODE_NAME, groupDescription=DEFAULT_NODE_NAME):
	"""
		于初始化一个HDF5文件，并返回tables.File类的实例，该类是用于处理HDF5文件的PyTables库的一部分
		toDiskName: the name of the file on disk
		toDiskName：HDF5文件的名称，它将被创建或覆盖。这是必需的参数。
		groupName：HDF5文件的根组的名称。此参数是可选的，默认为DEFAULT_NODE_NAME。
		groupDescription：根组的简短描述。此参数也是可选的，默认为DEFAULT_NODE_NAME
	"""
	##import tables;
	h5file = tables.open_file(toDiskName, mode="w", title="Dataset")
	gcolumns = h5file.createGroup(h5file.root, groupName, groupDescription)
	return h5file


class InfoToInitArrayOnH5File(object):
	def __init__(self, name, shape, atomicType):
		"""
			name: the name of this matrix
			shape: tuple indicating the shape of the matrix (similar to numpy shapes)
			atomicType: one of the pytables atomic types - eg: tables.Float32Atom() or tables.StringAtom(itemsize=length);
			这是一个Python类，用于存储一个矩阵的信息，方便在HDF5文件中初始化该矩阵。该类有三个属性：
			name：矩阵的名称。
			shape：一个元组，指示矩阵的形状（类似于numpy中的形状）。
			atomicType：pytables原子类型之一的实例，例如：tables.Float32Atom()或tables.StringAtom(itemsize=length)。
			该类的实例可用于在HDF5文件中创建一个相应的数组，并指定其名称、形状和数据类型。
		"""
		self.name = name
		self.shape = shape
		self.atomicType = atomicType


def writeToDisk(theH5Column, whatToWrite, batch_size=5000):
	"""
		Going to write to disk in batches of batch_size
		这是一个Python函数，用于将数据写入HDF5文件中。该函数需要三个参数：

		theH5Column：写入数据的HDF5列（HDF5文件的数据类型之一）。
		whatToWrite：要写入的数据。
		batch_size：每次写入的批次大小。此参数是可选的，默认值为5000。
		该函数使用循环将数据批次写入HDF5文件中，以避免一次性写入大量数据而导致内存占用过高。具体来说，函数将数据分成批次，每批次大小为batch_size，然后逐个批次将数据写入HDF5文件中。
		注意，在循环中使用 h5file.flush() 会将缓存中的数据写入磁盘。可能是因为 h5file 未定义，所以此代码中的 h5file.flush() 会引发 NameError 异常。
	""" 
	data_size = len(whatToWrite)
	last = int(data_size / float(batch_size)) * batch_size
	for i in range(0, data_size, batch_size):
		stop = (i + data_size%batch_size if i >= last
				else i + batch_size)
		theH5Column.append(whatToWrite[i:stop])
		##h5file.flush()


def getH5column(h5file, columnName, nodeName=DEFAULT_NODE_NAME):
	"""这是一个Python函数，用于获取HDF5文件中的一列数据。该函数需要三个参数：

	h5file：HDF5文件对象，即
	tables.File
	类的实例。
	columnName：要获取的列的名称。
	nodeName：要获取列所在节点的名称。此参数是可选的，默认值为
	DEFAULT_NODE_NAME。
	该函数使用
	h5file.getNode()
	方法获取指定节点对象，并使用
	getattr()
	函数从该节点对象中获取指定名称的属性，即该节点中的一列数据。

	注意，此函数假设指定的节点名称为根节点的默认名称
	DEFAULT_NODE_NAME。如果节点名称不同，应相应更改
	nodeName
	参数的值。"""

	node = h5file.get_node('/', DEFAULT_NODE_NAME)
	return getattr(node, columnName)


def initColumnsOnH5File(h5file, infoToInitArraysOnH5File, expectedRows, nodeName=DEFAULT_NODE_NAME, complib='blosc', complevel=5):
	"""
		h5file: filehandle to the h5file, initialised with init_h5_file
		infoToInitArrayOnH5File: array of instances of InfoToInitArrayOnH5File
		expectedRows: this code is set up to work with EArrays, which can be extended after creation.
			(presumably, if your data is too big to fit in memory, you're going to have to use EArrays
			to write it in pieces). "sizeEstimate" is the estimated size of the final array; it
			is used by the compression algorithm and can have a significant impace on performance.
		nodeName: the name of the node being written to.
		complib: the docs seem to recommend blosc for compression...
		complevel: compression level. Not really sure how much of a difference this number makes...
		这是一个Python函数，用于在HDF5文件中初始化一些列数据。该函数需要五个参数：
		h5file：HDF5文件对象，即 tables.File 类的实例。
		infoToInitArraysOnH5File：包含要初始化的列的信息的 InfoToInitArrayOnH5File 对象列表。
		expectedRows：预期行数。此代码段是为了与可扩展数组（EArrays）一起使用，EArrays 在创建后可以进行扩展。expectedRows 是最终数组的预估大小。它用于压缩算法，并且可能会对性能产生重大影响。
		nodeName：要在其中初始化列的节点名称。此参数是可选的，默认值为 DEFAULT_NODE_NAME。
		complib：压缩库的名称。此代码段建议使用 blosc 进行压缩。
		complevel：压缩级别。此参数是可选的，默认值为5。
		该函数使用循环遍历 infoToInitArraysOnH5File 列表中的每个元素，并使用 h5file.createEArray() 方法创建一个可扩展数组（EArray）。finalShape 是创建 EArray 的形状，它将原始形状的第一个维度设置为 0，因为此维度将在之后进行扩展。

			注意，此代码段假定要初始化的节点名称为根节点的默认名称 DEFAULT_NODE_NAME。如果节点名称不同，应相应更改 nodeName 参数的值。
	"""
	gcolumns = h5file.getNode(h5file.root, nodeName)
	filters = tables.Filters(complib=complib, complevel=complevel)
	for infoToInitArrayOnH5File in infoToInitArraysOnH5File:
		finalShape = [0] #in an eArray, the extendable dimension is set to have len 0
		finalShape.extend(infoToInitArrayOnH5File.shape)
		h5file.createEArray(gcolumns, infoToInitArrayOnH5File.name, atom=infoToInitArrayOnH5File.atomicType
							, shape=finalShape, title=infoToInitArrayOnH5File.name #idk what title does...
							, filters=filters, expectedrows=expectedRows)



def load_ATOM_BOX():
	
	dataName = "data"
	dataShape = [4, 20, 20, 20] #arr describing the dimensions other than the extendable dim.描述了除可扩展区域外的其他维度。
	labelName = "label"
	labelShape = []

	all_Xtr=[]
	all_ytr=[]
	all_train_sizes=[]
	train_mean=numpy.zeros((4,20,20,20))
	total_train_size=0

	for part in range (0,1):

		filename_train = "./data/ATOM_CHANNEL_dataset/train_data_"+str(part+1)+".pytables"
		h5file_train = tables.open_file(filename_train, mode="r")
		dataColumn_train = getH5column(h5file_train, dataName)
		labelColumn_train = getH5column(h5file_train, labelName)
		Xtr=dataColumn_train[:]
		ytr=labelColumn_train[:]
		total_train_size+=Xtr.shape[0]
		train_mean += numpy.mean(Xtr, axis=0)
		
		all_train_sizes.append(Xtr.shape[0])
		all_Xtr.append(Xtr)
		all_ytr.append(ytr)

	
	mean = train_mean/6
	norm_Xtr = []
	for Xtr in all_Xtr:
		Xtr -= mean
		norm_Xtr.append(Xtr)


	# Due to memorry consideration and training speed, we only used 1/6 test data to get a sense of the general test error. 
	# We test the full test dataset separately after the training is completed.
	for part in range (0,1): 
		filename_test = "./data/ATOM_CHANNEL_dataset/test_data_"+str(part+1)+".pytables"
		h5file_test = tables.open_file(filename_test, mode="r")
		dataColumn_test = getH5column(h5file_test, dataName)
		labelColumn_test = getH5column(h5file_test, labelName)
		Xt=dataColumn_test[:]
		yt=labelColumn_test[:]
		Xt -= mean 

		if part == 0:
			norm_Xt = Xt
			all_yt = yt
		else:
			norm_Xt = numpy.concatenate((norm_Xt,Xt), axis=0)
			all_yt = numpy.concatenate((all_yt,yt), axis=0)
	   
	# Same considerations as the above for the test dataset, more val data can be used to tune the hyper-parameters if desired
	for part in range (0,1):
		filename_val = "./data/ATOM_CHANNEL_dataset/val_data_"+str(part+1)+".pytables"
		h5file_val = tables.open_file(filename_val, mode="r")
		dataColumn_val = getH5column(h5file_val, dataName)
		labelColumn_val = getH5column(h5file_val, labelName)
		Xv=dataColumn_val[:]
		yv=labelColumn_val[:]
		Xv -= mean
		
		if part == 0:
			norm_Xv = Xv
			all_yv = yv
		else:
			norm_Xv = numpy.concatenate((norm_Xv,Xv), axis=0)
			all_yv = numpy.concatenate((all_yv,yv), axis=0)

	all_examples=[norm_Xtr,norm_Xt,norm_Xv]
	all_labels=[all_ytr,all_yt,all_yv]
	return [all_examples, all_labels, all_train_sizes, norm_Xt.shape[0], norm_Xv.shape[0]]
"""该函数返回了一个包含5个元素的列表，具体为：

all_examples: 包含三个元素的列表，每个元素都是一个 numpy 数组，分别表示训练集、测试集和验证集的特征数据；
all_labels: 包含三个元素的列表，每个元素都是一个 numpy 数组，分别表示训练集、测试集和验证集的标签数据；
all_train_sizes: 包含训练集的大小，即每个子数据集的样本数量的列表；
norm_Xt.shape[0]: 测试集特征数据的样本数量；
norm_Xv.shape[0]: 验证集特征数据的样本数量。"""


def load_FEATURE():
	ID = 'FEATURE_SCOP_T4_train_beta'
	for part in range(0, 5):
		data_part = numpy.load("../data/FEATURE_dataset/FEATURE_train_X_"+str(part)+".dat")
		labels_part = numpy.load("../data/FEATURE_dataset/FEATURE_train_y_"+str(part)+".dat")
		if part == 0:
			data = data_part
			labels = labels_part
		else:
			data = numpy.concatenate((data, data_part), axis=0)
			labels = numpy.concatenate((labels, labels_part), axis=0)
	data_mean = numpy.mean(data,axis=0)
	Xv = numpy.load("../data/FEATURE_dataset/FEATURE_val_X.dat")
	yv = numpy.load("../data/FEATURE_dataset/FEATURE_val_y.dat")

	Xt = numpy.load("../data/FEATURE_dataset/FEATURE_test_X.dat")
	yt = numpy.load("../data/FEATURE_dataset/FEATURE_test_y.dat")

	Xv -= data_mean
	Xt -= data_mean
	data -= data_mean

	Xtr=data
	ytr=labels
	return [Xtr, ytr, Xt, yt, Xv, yv]
	

