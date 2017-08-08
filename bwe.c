#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define SRC 0
#define TAR 1
#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define MAX_ITEM 5
#define INCREMENT 1000
#define MINI_INCREMENT 10
#define PARAPHRASE_THRESHOLD 0
#define PROBABILITY_THRESHOLD 0.01
#define BI_COOC_THRESHOLD 0

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const int trans_hash_size = 30000000;
char *src_tar_name[2] = { "source\0", "target\0" };
typedef float real;                    // Precision of float numbers

// 目标函数开关 1为使用 0为不使用
int src_cooc = 1;
int tar_cooc = 1;
int bi_sim = 1;
int mono_sim = 0;
real weight_cooc = 0.1;
real weight_bi = 0.9;
real weight_mono=0;
real j=0, src_j1=0, tar_j1=0, src_j2=0, tar_j2=0, c1_src_j3=0, c1_tar_j3=0, c2_j3=0;//计算代价

typedef struct vocab_word {
	long long cn;			// 词频
	long long trans_index;	// 在翻译表中的索引
	char *word;				// 词面
}vocab_word;

// 复述元组，即一个英文词对应了多个相思意思的英文词
typedef struct paraphrase_unit {
	long long src_index;		// 复述中的源端在词表中的位置
	long long tar_index;		// 复述中的目标端在词表中的位置
	real weight;		// 复述权重
}paraphrase, *paraphrase_p;

// 双语共现元组
typedef struct bi_cooc_unit {
	long long src_index;		// 翻译中的源端在词表中的位置
	long long tar_index;		// 翻译中的目标端在词表中的位置
	int count;					// 翻译中源端词语目标端词共现的次数
}bi_cooc, *bi_cooc_p;

// 翻译元组
typedef struct trans_unit {
	int num_trans;
	int *trans_count;
	long long src_index;			// 源语言词表索引
	long long *trans_indexs;	// 目标语言词表索引
	real *trans_pro;			// 翻译概率	
	int max_trans;
}translation, *translation_p;

typedef struct language{
	int src_tar;					// SRC/TAR
	long long vocab_max_size;		// 词汇表的最大长度，可以扩增，每次扩大1000
	long long vocab_size;			// 词汇的现有长度，接近vocab_max_size的时候会扩容
	long long para_max_size;		// 复述表的最大长度，可以扩增，每次扩大1000
	long long para_size;			// 复述的现有长度，接近para_max_size的时候会扩容
	long long trans_max_size;		// 翻译表的最大长度，可以扩增，每次扩大1000
	long long trans_size;			// 翻译表的现有长度，接近para_max_size的时候会扩容
	long long train_words;			// 训练的单词总数
	long long word_count_actual;	// 已经训练完词的个数
	vocab_word *vocab;				// 词表
	paraphrase *paras;				// 复述表
	translation *trans;				// 翻译表
	real *syn0;						// 模型产生的词向量
	real *syn1neg;					// 负采样产生的词向量
	int *vocab_hash;				// 词汇哈希表 下标是词的哈希，内容是该词在vocab中的位置 a[word_hash] = word index in vocab
	int *trans_hash;				// 翻译哈希表 下标是词的哈希，内容是该词在trans中的位置 a[word_hash] = word index in trans
	int *sample_table;				// 词采样表
	char train_file[MAX_STRING];	// 训练文件
	char output_file[MAX_STRING];	// 输出文件
	char pre_file[MAX_STRING];	// 输出文件
	char save_vocab_file[MAX_STRING];	// 保存词表文件
	char read_vocab_file[MAX_STRING];	// 读取词表文件
	char para_file[MAX_STRING];	// 读取复述文件
	int file_size;					//训练文件大小
}language, *language_p;

// 初始化语言结构体
void init_language(language_p lang_p, int src_tar)
{
	lang_p->src_tar = src_tar;
	lang_p->vocab_max_size = INCREMENT;
	lang_p->vocab_size = 0;
	lang_p->para_max_size = INCREMENT;
	lang_p->para_size = 0;
	lang_p->trans_max_size = INCREMENT;
	lang_p->trans_size = 0;
	lang_p->train_words = 0;
	lang_p->word_count_actual = 0;
	lang_p->vocab = (vocab_word *)calloc(lang_p->vocab_max_size, sizeof(vocab_word));
	lang_p->paras = (paraphrase *)calloc(lang_p->para_max_size, sizeof(paraphrase));
	// TODO
	lang_p->trans = (translation *)calloc(lang_p->trans_max_size, sizeof(translation));
	lang_p->syn0 = NULL;
	lang_p->syn1neg = NULL;
	lang_p->vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
	lang_p->sample_table = NULL;
	lang_p->train_file[0] = 0;
	lang_p->output_file[0] = 0;
	lang_p->pre_file[0] = 0;
	lang_p->save_vocab_file[0] = 0;
	lang_p->read_vocab_file[0] = 0;
	lang_p->para_file[0] = 0;
	lang_p->file_size = 0;
}

// 定义两种语言
language_p src_lang, tar_lang;

/*
 * binary		是否二进制输出
 * debug_mode	调试模型
 * window		skip-gram model窗口大小
 * min_count	词过滤阈值 词计数
 * num_threads	线程数
 * min_reduce	
 */
int binary = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;

/*
 * src_vocab_hash	词汇哈希表
 * 下标是词的哈希，内容是该词在vocab中的位置
 * a[word_hash] = word index in vocab
 */
// int *src_vocab_hash;
// int *tar_vocab_hash;

/*
 * vocab_max_size	词汇表的最大长度，可以扩增，每次扩大1000
 * vocab_size		词汇的现有长度，接近vocab_max_size的时候会扩容
 * layer1_size		词向量维度，隐层的节点数
 */
// long long src_vocab_max_size = 1000, src_vocab_size = 0;
// long long tar_vocab_max_size = 1000, tar_vocab_size = 0;
long long layer1_size = 100;

/*
 * train_words			训练的单词总数
 * word_count_actual	已经训练完词的个数
 * iter					迭代次数
 * file_size			文件大小 TODO 替换成文件行数
 * classes				输出cluster个数
 */
// long long src_train_words = 0, tar_train_words = 0;
// long long src_word_count_actual = 0, tar_word_count_actual = 0;
long long train_lines = 0, line_count_actual = 0, iter = 5, classes = 0;

/*
 * alpha			学习率
 * starting_alpha	初始alpha值
 * sample			亚采样概率的参数，0代表不进行亚采样
 */
real alpha = 0.025, starting_alpha, sample = 1e-3;

/*
 * syn0			模型产生的词向量
 * syn1neg		Neg Sampling 产生的词向量
 * expTable		预先计算Exp值
 */
real *expTable; // *src_syn0, *src_syn1neg
// real *tar_syn0, *tar_syn1neg;
clock_t start;

int hs = 0, negative = 5;
/*
 * table_size	词采样表的大小
 * table		词采样表
 */
const int table_size = 1e8;


char bi_cooc_file[MAX_STRING] = { 0 };
long long bi_cooc_size = 0;
long long bi_cooc_max_size = INCREMENT * 10;
bi_cooc *bi_cooc_table = NULL;
int *bi_cooc_sample_table = NULL;

// 将str字符以spl分割,存于dst中，并返回子字符串数量
/*
	char str2[] = "wh ||| is ||| y";
	char dst[10][80];
	int cnt = split(dst, str, " ||| ");
	int i = 0;
	for (i = 0; i < cnt; i++)
		puts(dst[i]);
*/
int split(char dst[][MAX_STRING * 2], char* str, const char* spl)
{
	int n = 0;
	char *result = NULL;
	result = strtok(str, spl);//把一个字符串分解为一串字符串
	while (result != NULL)
	{
		strcpy(dst[n++], result);
		result = strtok(NULL, spl);
	}
	return n;
}

/* 重新计算目标函数的权重 */
void RecalcWeight() {
    real sum = weight_cooc;
    sum += bi_sim? weight_bi: 0;
    sum += (mono_sim == 1 || mono_sim == 2)? weight_mono: 0;
    weight_cooc /= sum;
    weight_bi /= sum;
    weight_mono /= sum;
}

/*
 * 初始化词采样表
 * 下标是词采样表的位置 内容是词在词表中的位置
 * table[i] = word index in vocab
 */
// SRC TAR
void InitUnigramTable(language_p lang_p) {
	int a, i;
	double train_words_pow = 0;
	double d1, power = 0.75;

	lang_p->sample_table = (int *)malloc(table_size * sizeof(int));
	for (a = 0; a < lang_p->vocab_size; a++) train_words_pow += pow((double)(lang_p->vocab)[a].cn, power);
	i = 0;
	d1 = pow((double)(lang_p->vocab)[i].cn, power) / train_words_pow;
	for (a = 0; a < table_size; a++) {
		lang_p->sample_table[a] = i;
		if (a / (double)table_size > d1) {
			i++;
			d1 += pow((double)(lang_p->vocab)[i].cn, power) / train_words_pow;//按概率分布词采样表
		}
		if (i >= lang_p->vocab_size) i = lang_p->vocab_size - 1;
	}
}

void InitBicoocSampleTable() {
	int a, i;
	double train_words_pow = 0;
	double d1, power = 0.75;
	// FILE *fout = NULL;
	// fout = fopen("sample_table.txt", "w");
	bi_cooc_sample_table = (int *)malloc(table_size * sizeof(int));
	for (a = 0; a < bi_cooc_size; a++) train_words_pow += pow(bi_cooc_table[a].count, power);
	i = 0;
	d1 = pow(bi_cooc_table[i].count, power) / train_words_pow;
	for (a = 0; a < table_size; a++) {
		bi_cooc_sample_table[a] = i;
		// fprintf(fout, "%d %d %d\n", a, bi_cooc_sample_table[a], i);
		if (a / (double)table_size > d1) {
			i++;
			d1 += pow(bi_cooc_table[i].count, power) / train_words_pow;
		}
		if (i >= bi_cooc_size) i = bi_cooc_size - 1;
	}
	// fclose(fout);
	printf("InitBicoocSampleTable over!\n");
}

void InitTransPro(language_p lang_p) {
	long long c = 0, d = 0;
	int sum = 0;
	for (c = 0; c < lang_p->trans_size; c++) {
		sum = 0;
		lang_p->trans[c].trans_pro = (real *)calloc(lang_p->trans[c].num_trans, sizeof(real));
		for (d = 0; d < lang_p->trans[c].num_trans; d++)
			sum += lang_p->trans[c].trans_count[d];
		for (d = 0; d < lang_p->trans[c].num_trans; d++)
			lang_p->trans[c].trans_pro[d] = (real)lang_p->trans[c].trans_count[d] / sum; 
		free(lang_p->trans[c].trans_count);	
	}

	printf("InitTransPro success!\n");
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
// 每次从文件中读取一个词
// 空格、制表、换行为分隔符
// SRC TAR
void ReadWord(char *word, FILE *fin) {
	int a = 0, ch;
	while (!feof(fin)) {
		ch = fgetc(fin);
		if (ch == 13) continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
			if (a > 0) {
				if (ch == '\n') ungetc(ch, fin);
				break;
			}
			if (ch == '\n') {
				strcpy(word, (char *)"</s>");
				return;
			}
			else continue;
		}
		word[a] = ch;
		a++;
		if (a >= MAX_STRING - 1) a--;   // Truncate too long words
	}
	word[a] = 0;
}

// Returns hash value of a word
// 查找词的哈希值
// SRC TAR
int GetWordHash(char *word) {
	unsigned long long a, hash = 0;
	for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
	hash = hash % vocab_hash_size;
	return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
// 查找词在词表中的位置
// SRC TAR
int SearchVocab(char *word, language_p lang_p) {
	unsigned int hash = GetWordHash(word);
	while (1) {
		if (lang_p->vocab_hash[hash] == -1) return -1;
		if (!strcmp(word, (lang_p->vocab[lang_p->vocab_hash[hash]]).word))
			return lang_p->vocab_hash[hash];
		hash = (hash + 1) % vocab_hash_size;
	}
	return -1;
}

/*// Returns position of a word in the Translate Table; if the word is not found, returns -1
// 查找词在翻译表中的位置
// SRC TAR
int SearchTrans(char *word, language_p lang_p) {
	long long index = SearchVocab(word, lang_p);
	unsigned int hash = 0;
	if (index == -1)	// 查无此词
		return -1;
	hash = GetWordHash(word);
	while (1) {
		if (lang_p->trans_hash[hash] == -1) return -1;
		if ((lang_p->trans[lang_p->trans_hash[hash]]).src_index == index)
			return lang_p->trans_hash[hash];
		hash = (hash + 1) % trans_hash_size;
	}
	return -1;
}*/

// Reads a word and returns its index in the vocabulary
// 从文件中读入一个词，直接返回其在词表中的位置
// SRC TAR
int ReadWordIndex(FILE *fin, language_p lang_p) {
	char word[MAX_STRING];
	ReadWord(word, fin);
	if (feof(fin)) return -1;
	return SearchVocab(word, lang_p);
}

// Adds a word to the vocabulary
// 在词表中添加一个词
// SRC TAR
int AddWordToVocab(char *word, language_p lang_p) {
	unsigned int hash, length = strlen(word) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	(lang_p->vocab[lang_p->vocab_size]).word = (char *)calloc(length, sizeof(char));
	strcpy((lang_p->vocab[lang_p->vocab_size]).word, word);
	(lang_p->vocab[lang_p->vocab_size]).cn = 0;
	(lang_p->vocab[lang_p->vocab_size]).trans_index = -1;
	lang_p->vocab_size++;
	// Reallocate memory if needed
	if (lang_p->vocab_size + 2 >= lang_p->vocab_max_size) {
		lang_p->vocab_max_size += INCREMENT;
		lang_p->vocab = (struct vocab_word *)realloc(lang_p->vocab, lang_p->vocab_max_size * sizeof(struct vocab_word));
	}
	hash = GetWordHash(word);
	while (lang_p->vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
	lang_p->vocab_hash[hash] = lang_p->vocab_size - 1;
	return lang_p->vocab_size - 1;
}

// ERROR
// Adds a word to the trans
// 在翻译表中添加一个词
/*
 * index	词表中索引		long long
 * lang_p	语言类型指针	language_p
 */
// SRC TAR
int AddWordToTrans(long long index, language_p lang_p) {
	if (index == -1) // 查无此词
		return -1;
	lang_p->trans[lang_p->trans_size].src_index = index;
	lang_p->trans[lang_p->trans_size].num_trans = 0;
	lang_p->trans[lang_p->trans_size].max_trans = MINI_INCREMENT;
	lang_p->trans[lang_p->trans_size].trans_indexs = (long long *)calloc(MINI_INCREMENT, sizeof(long long));
	lang_p->trans[lang_p->trans_size].trans_pro = NULL;
	lang_p->trans[lang_p->trans_size].trans_count = (int *)calloc(MINI_INCREMENT, sizeof(int));
	lang_p->vocab[index].trans_index = lang_p->trans_size;
	lang_p->trans_size++;
	// Reallocate memory if needed
	if (lang_p->trans_size + 2 >= lang_p->trans_max_size) {
		lang_p->trans_max_size += INCREMENT;
		lang_p->trans = (translation *)realloc(lang_p->trans, lang_p->trans_max_size * sizeof(translation));
	}
	return lang_p->trans_size - 1;
}

// Adds a word to the trans
// 在翻译表中添加一个词
/*
 * src_trans_index	源语言翻译表中索引		long long
 * tar_index		目标语言词表中索引		long long
 * count			双语计数				int
 * src_p			语言类型指针	language_p
 */
// SRC TAR
int AddTarToSrc(long long src_trans_index, long long tar_index, int count, language_p src_p) {
	translation_p src_trans_unit_p;
	int num_trans = 0;
	src_trans_unit_p = &(src_p->trans[src_trans_index]);
	num_trans = src_trans_unit_p->num_trans;
	src_trans_unit_p->trans_count[num_trans] = count;
	src_trans_unit_p->trans_indexs[num_trans] = tar_index;
	src_trans_unit_p->num_trans++;
	
	// Reallocate memory if needed
	if (src_trans_unit_p->num_trans + 2 >= src_trans_unit_p->max_trans) {
		src_trans_unit_p->max_trans += MINI_INCREMENT;
		src_trans_unit_p->trans_count = (int*)realloc(src_trans_unit_p->trans_count, src_trans_unit_p->max_trans * sizeof(int));
		src_trans_unit_p->trans_indexs = (long long *)realloc(src_trans_unit_p->trans_indexs, src_trans_unit_p->max_trans * sizeof(long long));
	}
	return src_trans_unit_p->num_trans;
}

int AddTransRuleToTrans(long long src_index, long long tar_index, int count) {
	long long src_trans_index = 0, tar_trans_index = 0;
	if (src_index == -1 || tar_index == -1)
		return -1;
	src_trans_index = src_lang->vocab[src_index].trans_index;
	tar_trans_index = tar_lang->vocab[tar_index].trans_index;
	if (src_trans_index == -1)	//	 若还无翻译规则，则在翻译表中添加
		src_trans_index = AddWordToTrans(src_index, src_lang);
	if (tar_trans_index == -1)	//	 若还无翻译规则，则在翻译表中添加
		tar_trans_index = AddWordToTrans(tar_index, tar_lang);
	if (src_trans_index == -1 || tar_trans_index == -1)
		return -1;
	AddTarToSrc(src_trans_index, tar_index, count, src_lang);
	AddTarToSrc(tar_trans_index, src_index, count, tar_lang);
	return 1;
}

// Used later for sorting by word counts
// 词频大小比较
// SRC TAR
int VocabCompare(const void *a, const void *b) {
	return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;  //越在前面词频越小
}

// Sorts the vocabulary by frequency using word counts
// 依据词频对词表排序
// SRC
void SortVocab(language_p lang_p) {
	int a, size;
	unsigned int hash;
	// Sort the vocabulary and keep </s> at the first position
	qsort(&(lang_p->vocab[1]), lang_p->vocab_size - 1, sizeof(struct vocab_word), VocabCompare);//排序的时候把<s>,</s>排除开
	for (a = 0; a < vocab_hash_size; a++) lang_p->vocab_hash[a] = -1;
	size = lang_p->vocab_size;
	lang_p->train_words = 0;
	for (a = 0; a < size; a++) {
		// Words occuring less than min_count times will be discarded from the vocab
		if (((lang_p->vocab[a]).cn < min_count) && (a != 0)) {//a=0的时候为<s>
			lang_p->vocab_size--;
			free((lang_p->vocab[a]).word);
		}
		else {
			// Hash will be re-computed, as after the sorting it is not actual
			hash = GetWordHash((lang_p->vocab[a]).word);
			while (lang_p->vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
			lang_p->vocab_hash[hash] = a;
			lang_p->train_words += (lang_p->vocab[a]).cn;
		}
	}
	lang_p->vocab = (struct vocab_word *)realloc(lang_p->vocab, (lang_p->vocab_size + 1) * sizeof(struct vocab_word));
}

// 保存词表
// SRC TAR
void SaveVocab(language_p lang_p) {
	long long i;
	FILE *fo = fopen(lang_p->save_vocab_file, "wb");
	for (i = 0; i < lang_p->vocab_size; i++)
		fprintf(fo, "%s %lld\n", (lang_p->vocab[i]).word, (lang_p->vocab[i]).cn);
	fclose(fo);
}

// 读取词表
// SRC
void ReadVocab(language_p lang_p) {
	long long a, i = 0;
	char c;
	char word[MAX_STRING];
	FILE *fin = NULL;
	fin = fopen(lang_p->read_vocab_file, "rb");
	if (fin == NULL) {
		printf("Vocabulary file not found\n");
		exit(1);
	}
	for (a = 0; a < vocab_hash_size; a++) lang_p->vocab_hash[a] = -1;
	lang_p->vocab_size = 0;
	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;
		a = AddWordToVocab(word, lang_p);
		fscanf(fin, "%lld%c", &(lang_p->vocab[a]).cn, &c);  //词汇表文件的格式是词  词频的形式
		i++;
	}
	SortVocab(lang_p);
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", lang_p->vocab_size);
		printf("Sentences in train file: %lld\n", train_lines);
	}
	fin = fopen(lang_p->train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}

	fseek(fin, 0, SEEK_END);
	lang_p->file_size = ftell(fin);
	fclose(fin);
}

// Reduces the vocabulary by removing infrequent tokens
// 在词表中删去低频词 提高哈希效率
// SRC TAR
void ReduceVocab(language_p lang_p) {
	int a, b = 0;
	unsigned int hash;
	for (a = 0; a < lang_p->vocab_size; a++) 
		if ((lang_p->vocab[a]).cn > min_reduce) {
			(lang_p->vocab[b]).cn = (lang_p->vocab[a]).cn;
			(lang_p->vocab[b]).word = (lang_p->vocab[a]).word;
			b++;
		}
	else free((lang_p->vocab[a]).word);
	lang_p->vocab_size = b;
	for (a = 0; a < vocab_hash_size; a++) 
		lang_p->vocab_hash[a] = -1;
	for (a = 0; a < lang_p->vocab_size; a++) {
		// Hash will be re-computed, as it is not actual
		hash = GetWordHash((lang_p->vocab[a]).word);
		while (lang_p->vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
		lang_p->vocab_hash[hash] = a;
	}
	fflush(stdout);
	min_reduce++;
}

// 从训练文件中读取词表
// SRC TAR
void LearnVocabFromTrainFile(language_p lang_p) {
	char word[MAX_STRING];
	FILE *fin;
	long long a, i;
	for (a = 0; a < vocab_hash_size; a++) lang_p->vocab_hash[a] = -1;
	fin = fopen(lang_p->train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	lang_p->vocab_size = 0;
	AddWordToVocab((char *)"</s>", lang_p);
	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;
		lang_p->train_words++;
		if ((debug_mode > 1) && (lang_p->train_words % 100000 == 0)) {
			printf("%lldK%c", lang_p->train_words / 1000, 13);
			fflush(stdout);
		}
		i = SearchVocab(word, lang_p);
		if (i == -1) {
			a = AddWordToVocab(word, lang_p);
			(lang_p->vocab[a]).cn = 1;
		}
		else (lang_p->vocab[i]).cn++;
		if (lang_p->vocab_size > vocab_hash_size * 0.7) ReduceVocab(lang_p);
	}
	SortVocab(lang_p);
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", lang_p->vocab_size);
		printf("Words in train file: %lld\n", lang_p->train_words);
	}
	lang_p->file_size = ftell(fin);
	fclose(fin);
}

// 保存词向量
// SRC TAR
void SaveVectors(language_p lang_p) {
	FILE *fo;
	int a, b;
	fo = fopen(lang_p->output_file, "wb");
	fprintf(fo, "%lld %lld\n", lang_p->vocab_size, layer1_size);
	for (a = 0; a < lang_p->vocab_size; a++) {
		fprintf(fo, "%s ", lang_p->vocab[a].word);
		if (binary) for (b = 0; b < layer1_size; b++) fwrite(&(lang_p->syn0[a * layer1_size + b]), sizeof(real), 1, fo);
		else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", lang_p->syn0[a * layer1_size + b]);
		fprintf(fo, "\n");
	}
	fclose(fo);
}

real GetPraPro(language_p lang_p,int src_index,int tar_index)
{
	int a;
	int trans_index;
	translation_p trans;
	//int trans_index=(lang_p->vocab)[src_index].trans_index;
	//translation trans=(lang_p->trans)[trans_index];
	//printf("2\n");
	trans_index=(lang_p->vocab)[src_index].trans_index;
	trans=&(lang_p->trans[trans_index]);
	//printf("3\n");
	for(a=0;a<trans->num_trans;a++)
		if(trans->trans_indexs[a]==tar_index)
			return trans->trans_pro[a];
	return 0;
}

real CountSingleParaWeight(language_p lang_p,int src_index,int tar_index)
{
	language_p lang_p2;
	real sum=0;
	real pro1=0,pro2=0;
	int a;
	int trans_index=((lang_p->vocab)[src_index]).trans_index;
	if (trans_index==-1)
		return -1;
	if(lang_p->src_tar==0)
		lang_p2=tar_lang;
	else lang_p2=src_lang;
	//printf("1\n");
	//printf("%d\n",trans_index);
	//printf("%f\n",lang_p->trans[trans_index].trans_pro[0]);
	for(a=0;a<(lang_p->trans)[trans_index].num_trans;a++)
	{
		pro1=lang_p->trans[trans_index].trans_pro[a];
		pro2=GetPraPro(lang_p2,((lang_p->trans)[trans_index].trans_indexs)[a],tar_index);
		sum+=pro1*pro2;

	}
	return sum;
}

// 从复述文件中读取复述
// SRC TAR
void ReadParaPhrase(language_p lang_p) {//格式为源词下标，目标词下表，权重
	FILE *fin = NULL;
	char StrLine[MAX_STRING * 2], items[MAX_ITEM][MAX_STRING * 2];
	long long split_num = 0, line = 0, discard = 0, src_index = 0, tar_index = 0;
	real weight = 0;
	real pro1,pro2;
	fin = fopen(lang_p->para_file, "r");
	if (fin == NULL) {
		printf("ERROR: %s paraphrase file not found!\n", src_tar_name[lang_p->src_tar]);
		exit(1);
	}
	while (1) {
		line += 1;
		StrLine[0] = 0;
		fgets(StrLine, MAX_STRING * 2, fin);  //读取一行
		split_num = split(items, StrLine, " ||| ");
		if (feof(fin)) break;
		if (split_num < 3) {
			printf("Warning: illegal line %lld in %s: %s", line, lang_p->para_file, StrLine);
			continue;
		}
		src_index = SearchVocab(items[0], lang_p);
		tar_index = SearchVocab(items[1], lang_p);
		weight = atof(items[2]);
		if (src_index == -1 || tar_index == -1 || weight < PARAPHRASE_THRESHOLD) {
			discard++;
			continue;
		}
		(lang_p->paras)[lang_p->para_size].src_index = src_index;
		(lang_p->paras)[lang_p->para_size].tar_index = tar_index;
		(lang_p->paras)[lang_p->para_size].weight = weight;
		//pro1=CountSingleParaWeight(lang_p,src_index,tar_index);
		//pro2=CountSingleParaWeight(lang_p,tar_index,src_index);
		//if(pro1==-1||pro2==-1)
			//(lang_p->paras)[lang_p->para_size].weight = weight;
		//else (lang_p->paras)[lang_p->para_size].weight =(pro1+pro2)/2;
		lang_p->para_size++;
		// Reallocate memory if needed
		if (lang_p->para_size + 2 >= lang_p->para_max_size) {
			lang_p->para_max_size += INCREMENT;
			lang_p->paras = (paraphrase *)realloc(lang_p->paras, lang_p->para_max_size * sizeof(paraphrase));
		}
		if ((debug_mode > 1) && (lang_p->para_size % 100000 == 0)) {
			printf("%lldK%c", lang_p->para_size / 1000, 13);
			fflush(stdout);
		}
	}
	printf("Paraphrase rules in %s paraphrase file: %lld, ", src_tar_name[lang_p->src_tar], lang_p->para_size);
	printf("and %lld discard.\n", discard);
	fclose(fin);                     //关闭文件
}

// 从翻译计数文件中读取翻译计数
// SRC TAR
void ReadTransCount() {
	FILE *fin = NULL;
	char StrLine[MAX_STRING * 2], items[MAX_ITEM][MAX_STRING * 2];
	long long split_num = 0, line = 0, discard = 0, src_index = 0, tar_index = 0;
	int count = 0;
	fin = fopen(bi_cooc_file, "r");
	if (fin == NULL) {
		printf("ERROR: %s bilingual co-occurrence file not found!\n", bi_cooc_file);
		exit(1);
	}
	bi_cooc_table = (bi_cooc *)calloc(bi_cooc_max_size, sizeof(bi_cooc));
	while (1) {
		line += 1;
		StrLine[0] = 0;
		fgets(StrLine, MAX_STRING * 2, fin);  //读取一行
		split_num = split(items, StrLine, " ");
		if (feof(fin)) break;
		if (split_num < 3) {
			printf("Warning: illegal line %lld in %s: %s", line, bi_cooc_file, StrLine);
			continue;
		}
		src_index = SearchVocab(items[0], src_lang);
		tar_index = SearchVocab(items[1], tar_lang);
		count = atoll(items[2]);
		if (src_index == -1 || tar_index == -1 || count < BI_COOC_THRESHOLD) {
			discard++;
			continue;
		}
		bi_cooc_table[bi_cooc_size].src_index = src_index;
		bi_cooc_table[bi_cooc_size].tar_index = tar_index;
		bi_cooc_table[bi_cooc_size].count = count;
		bi_cooc_size++;
		// 添加翻译规则
		AddTransRuleToTrans(src_index, tar_index, count);
		// Reallocate memory if needed
		if (bi_cooc_size + 2 >= bi_cooc_max_size) {
			bi_cooc_max_size += INCREMENT * 10;
			bi_cooc_table = (bi_cooc *)realloc(bi_cooc_table, bi_cooc_max_size * sizeof(bi_cooc));
		}
		if ((debug_mode > 1) && (bi_cooc_size % 100000 == 0)) {
			printf("%lldK%c", bi_cooc_size / 1000, 13);
			fflush(stdout);
		}
	}
	printf("Co-occurrenc Pairs in %s bilingual co-occurrence file: %lld, ", bi_cooc_file, bi_cooc_size);
	printf("and %lld discard.\n", discard);
	fclose(fin);                     //关闭文件
}

// 从预训练词向量文件中读取词向量
// SRC TAR
void LoadPreTrain(language_p lang_p) {//如果存在文件的话，其格式是第一行词个数，隐层单元格数;之后每一行一个单词，下一行为词向量
	FILE *fin = NULL;
	long long pre_train = 0, index = 0, b = 0, c = 0;
	long long size, words;
	char *word = (char *)calloc(MAX_STRING, sizeof(char));
	real *pre = (real *)calloc(layer1_size, sizeof(real));
	fin = fopen(lang_p->pre_file, "r");
	if (fin == NULL) {
		printf("ERROR: %s pre-train file %s not found!\n", src_tar_name[lang_p->src_tar], lang_p->pre_file);
		exit(1);
	}
	fscanf(fin, "%lld", &words);
	fscanf(fin, "%lld", &size);
	if (size != layer1_size) {
		printf("ERROR: %s pre-train file %s Dim ERROR!\n", src_tar_name[lang_p->src_tar], lang_p->pre_file);
		exit(1);
	}	
	for (b = 0; b < words; b++) {
		c = 0;
		while (1) {
			word[c] = fgetc(fin);
			if (feof(fin) || (word[c] == ' ')) break;
			if ((c < MAX_STRING) && (word[c] != '\n')) c++;
		}
		word[c] = 0;
		index = SearchVocab(word, lang_p);
		if (index == -1) {
			continue;
		}
		for (c = 0; c < layer1_size; c++)
			fscanf(fin, "%f ", &(pre[c]));
		for (c = 0; c < layer1_size; c++)
			lang_p->syn0[index * layer1_size + c] = pre[c];
		pre_train++;
	}
	printf("%lld words in %s pre-train file %s", pre_train, src_tar_name[lang_p->src_tar], lang_p->pre_file);
	fclose(fin);                     //关闭文件
	free(pre);
	free(word);
}

// 初始化词向量
// SRC TAR
void InitNet(language_p lang_p) {
	long long a, b;
	unsigned long long next_random = 1;
	a = posix_memalign((void **)&lang_p->syn0, 128, (long long)(lang_p->vocab_size) * layer1_size * sizeof(real));//分配的地址内存对齐
	if (lang_p->syn0 == NULL) { printf("Memory allocation failed\n"); exit(1); }
	if (negative>0) {
		a = posix_memalign((void **)&(lang_p->syn1neg), 128, (long long)(lang_p->vocab_size) * layer1_size * sizeof(real));
		if (lang_p->syn1neg == NULL) { printf("Memory allocation failed\n"); exit(1); }
		for (a = 0; a < lang_p->vocab_size; a++) for (b = 0; b < layer1_size; b++)
			lang_p->syn1neg[a * layer1_size + b] = 0;
	}
	for (a = 0; a < lang_p->vocab_size; a++) for (b = 0; b < layer1_size; b++) {
		next_random = next_random * (unsigned long long)25214903917 + 11;
		lang_p->syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
	}
}

// 读取一个句子
/*
* lang_p			训练的语言类型		language指针
* sen				训练实例所在的句子	整型数组
* sen_pos_p			句子位置指针		整型指针 为了与调用函数同步
* sen_len_p			句子长度指针		整型指针 为了与调用函数同步 句子训练完 长度变为0
* fin				文件指针			文件指针
* word_count_p		词计数指针			整型指针 为了与调用函数同步
* next_random_p		随机数指针			整型指针 为了与调用函数同步
*/
// SRC TAR
void ReadSentence(language_p lang_p, long long *sen, long long *sen_pos_p, long long *sen_len_p,
				  FILE *fin,  long long *word_count_p, unsigned long long *next_random_p)
{
	long long word;
	// 读取一个句子
	if ((*sen_len_p) == 0) {
		while (1) {
			word = ReadWordIndex(fin, lang_p);
			if (feof(fin)) break;
			if (word == -1) continue;
			(*word_count_p)++;
			if (word == 0) break;
			// The subsampling randomly discards frequent words while keeping the ranking same
			// 亚采样 减少高频词的训练
			if (sample > 0) {
				real ran = (sqrt((double)(lang_p->vocab[word]).cn / (sample * lang_p->train_words)) + 1) * (sample * lang_p->train_words) / (lang_p->vocab[word]).cn;
				(*next_random_p) = (*next_random_p) * (unsigned long long)25214903917 + 11;
				if (ran < ((*next_random_p) & 0xFFFF) / (real)65536) continue;
			}
			sen[(*sen_len_p)] = word;
			(*sen_len_p)++;
			if ((*sen_len_p) >= MAX_SENTENCE_LENGTH) break;
		}
		(*sen_pos_p) = 0;
	}
}

real get_log(real x)
{
	return log(x);
}

real get_sigmoid(real x)
{
	return 1/(1+exp(-x));
}

real fast_sigmoid(real x) {
	return expTable[(int)((x + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
}

void NegativeSample(real *pos, real *neg, real *neu1e, int label, real weight_obj) {
	long long c = 0;
	real f = 0, g = 0;
	for (c = 0; c < layer1_size; c++) f += pos[c] * neg[c];
	if (f > MAX_EXP) g = (label - 1) * alpha;
	else if (f < -MAX_EXP) g = (label - 0) * alpha;
	else g = (label - fast_sigmoid(f)) * alpha;
	// 计算反向传播的误差
	for (c = 0; c < layer1_size; c++) neu1e[c] += g * neg[c];
	// 更新负采样词向量
	for (c = 0; c < layer1_size; c++) neg[c] += g * pos[c]*weight_obj;
	/*
	long long c = 0;
    real f = 0, g = 0;
    for (c = 0; c < layer1_size; c++) f += pos[c] * neg[c];
    g = (label - fast_sigmoid(f)) * alpha;;
    // 计算反向传播的误差
    for (c = 0; c < layer1_size; c++) neu1e[c] += g * neg[c];
    // 更新负采样词向量
    for (c = 0; c < layer1_size; c++) neg[c] += g * pos[c] * weight_obj;*/
}

// 训练一个SGNS的实例
/* 
 * lang_p			训练的语言类型		language指针
 * sen				训练实例所在的句子	整型数组
 * sen_pos_p		句子位置指针		整型指针 为了与调用函数同步
 * sen_len_p		句子长度指针		整型指针 为了与调用函数同步 句子训练完 长度变为0
 * next_random_p	随机数指针			整型指针 为 了与调用函数同步
 * neu1e            误差累加			real数组 减少函数内重复申请内存
 */
// SRC TAR
void TrainOneSGNS(language_p lang_p, long long *sen, long long *sen_pos_p, long long *sen_len_p,
				  unsigned long long *next_random_p, real *neu1e) {
	long long a, b, c, d;
	// 在词向量表中的偏移量
	// l1为词向量偏移量
	// l2为负采样词向量偏移量
	long long l1, l2,l3,l4;
	long long target, word, last_word;
	int label;
	real y=0,y1=0;
	// 取一个词进行训练
	word = sen[(*sen_pos_p)];
	if (word == -1) return;
	for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
	// 动态窗口
	(*next_random_p) = (*next_random_p) * (unsigned long long)25214903917 + 11;
	b = (*next_random_p) % window;
	for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
		// 取出该词的窗口词
		c = (*sen_pos_p) - window + a;
		if (c < 0) continue;
		if (c >= (*sen_len_p)) continue;
		last_word = sen[c];
		if (last_word == -1) continue;
		l1 = last_word * layer1_size;
		for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
		l3=word*layer1_size;
	    y1=0
	    for(c=0;c<layer1_size;c++) y1+=lang_p->syn0[c + l1]*lang_p->syn0[c + l3];
	    //printf("1 ok\n");
	    y1=get_log(get_sigmoid(y1));
	    y+=y1;
	    //printf("2 ok\n");
	    //printf("3 ok\n");
		// NEGATIVE SAMPLING
		if (negative > 0) for (d = 0; d < negative + 1; d++) {
			if (d == 0) {
				target = word;
				label = 1;
			}
			else {
				(*next_random_p) = (*next_random_p) * (unsigned long long)25214903917 + 11;
				target = lang_p->sample_table[((*next_random_p) >> 16) % table_size];
				if (target == 0) target = (*next_random_p) % (lang_p->vocab_size - 1) + 1;
				if (target == word) continue;
				label = 0;
				l4=target*layer1_size;
				y1=0;
			    for(c=0;c<layer1_size;c++) y1-=lang_p->syn0[c + l3]*lang_p->syn0[c + l4];
			    //printf("4 ok\n");
			    y1=get_log(get_sigmoid(y1));
			    //printf("5 ok\n");
		        y+=y1;
		        //printf("6 ok\n");
			}
			l2 = target * layer1_size;
			
			/*
			f = 0;
			for (c = 0; c < layer1_size; c++) f += lang_p->syn0[c + l1] * lang_p->syn1neg[c + l2];
			if (f > MAX_EXP) g = (label - 1) * alpha;
			else if (f < -MAX_EXP) g = (label - 0) * alpha;
			else g = (label - fast_sigmoid(f)) * alpha;
			// 计算反向传播的误差
			for (c = 0; c < layer1_size; c++) neu1e[c] += g * lang_p->syn1neg[c + l2];
			// 更新负采样词向量
			for (c = 0; c < layer1_size; c++) lang_p->syn1neg[c + l2] += g * lang_p->syn0[c + l1];*/
			NegativeSample((lang_p->syn0) + l1, (lang_p->syn1neg) + l2, neu1e, label,weight_cooc);
		}
		// Learn weights input -> hidden
		// 更新词向量
		for (c = 0; c < layer1_size; c++) lang_p->syn0[c + l1] += neu1e[c]*weight_cooc;
	}
    if(lang_p->src_tar==0)
    	src_j1+=y;
    else tar_j1+=y;
	(*sen_pos_p)++;
	if ((*sen_pos_p) >= (*sen_len_p)) {
		(*sen_len_p) = 0;
		return;
	}
}


// 训练一个Sim Bi实例
void TrainSimBi(language_p lang_src_p, language_p lang_tar_p, real *neu1e, unsigned long long *next_random_p) {
	// 随机抽取一个src词
	long long c = 0, d = 0;
	long long src_index = 0, src_trans_index= 0 ;
	long long l1 = 0, l2 = 0;
	real weight = 0;
	real *y = (real *)calloc(layer1_size, sizeof(real));
	real yy=0;
	(*next_random_p) = (*next_random_p) * (unsigned long long)25214903917 + 11;
	src_trans_index = ((*next_random_p) >> 16) % lang_src_p->trans_size;
	src_index = lang_src_p->trans[src_trans_index].src_index;
	// 计算残差
	l1 = src_index * layer1_size;
	for (c = 0; c < layer1_size; c++) neu1e[c] = lang_src_p->syn0[l1 + c];
	for (c = 0; c < lang_src_p->trans[src_trans_index].num_trans; c++) {
		//l2 = lang_src_p->trans[src_trans_index].trans_indexs[c];  
		l2 =(lang_src_p->trans[src_trans_index].trans_indexs[c])*layer1_size;
		for(d=0;d<layer1_size;d++) y[d]=(lang_tar_p->syn0[l2+d])*(lang_src_p->trans[src_trans_index].trans_pro[c]);
		/*
		if (weight < PROBABILITY_THRESHOLD)   
			continue;
		weight = lang_src_p->trans[src_trans_index].trans_pro[c];*/
		weight = lang_src_p->trans[src_trans_index].trans_pro[c];
		if (weight < PROBABILITY_THRESHOLD)   
			continue;
		for (d = 0; d < layer1_size; d++) neu1e[d] -= weight * lang_tar_p->syn0[l2 + d];
	}

	// 更新源语言词向量
	for (c = 0; c < layer1_size; c++) lang_src_p->syn0[l1 + c] -= 2 * alpha * neu1e[c]*weight_bi;
	// 更新目标语言词向量
	for (c = 0; c < lang_src_p->trans[src_trans_index].num_trans; c++) {
		//l2 = lang_src_p->trans[src_trans_index].trans_indexs[c];           
		l2 =(lang_src_p->trans[src_trans_index].trans_indexs[c])*layer1_size; 
		weight = lang_src_p->trans[src_trans_index].trans_pro[c];
		if (weight < PROBABILITY_THRESHOLD)                 
			continue;
		for (d = 0; d < layer1_size; d++) lang_tar_p->syn0[l2 + d] += 2 * alpha * weight * neu1e[d]*weight_bi;
	}
    for(c=0;c<layer1_size;c++) yy+=(lang_src_p->syn0[l1 + c]-y[c])*(lang_src_p->syn0[l1 + c]-y[c]);
    if(lang_src_p->src_tar==0)
    	src_j2+=yy;
    else tar_j2+=yy;
    free(y);
}

// 训练一个Sim Mono 类型1 的实例
/*
* lang_p			训练的语言类型		language指针
* neu1e            误差累加			real数组 减少函数内重复申请内存
* next_random_p	随机数指针			整型指针 为 了与调用函数同步
*/
// SRC TAR
void TrainSimMonoType1(language_p lang_p, real *neu1e, unsigned long long *next_random_p) {
	long long c = 0;
	long long sim_mono_index = 0;			// 复述表的索引
	long long index = 0, para_index = 0;	// 复述表的src索引 tar索引
	real weight_para = 0;
	real y=0;
	(*next_random_p) = (*next_random_p) * (unsigned long long)25214903917 + 11;
	sim_mono_index = ((*next_random_p) >> 16) % lang_p->para_size;
	index = lang_p->paras[sim_mono_index].src_index * layer1_size;
	para_index = lang_p->paras[sim_mono_index].tar_index * layer1_size;
	weight_para = lang_p->paras[sim_mono_index].weight;
	if (index == -1 || para_index == -1 || weight_para < PARAPHRASE_THRESHOLD)
		return;
	// 更新词向量
	for (c = 0; c < layer1_size; c++) neu1e[c] = 2 * weight_para * (lang_p->syn0[index + c] - lang_p->syn0[para_index + c]); 
	for (c = 0; c < layer1_size; c++) lang_p->syn0[index + c] -= alpha * neu1e[c]*weight_mono;
	for (c = 0; c < layer1_size; c++) lang_p->syn0[para_index + c] += alpha * neu1e[c]*weight_mono;
	for(c=0;c<layer1_size;c++) y+=(lang_p->syn0[index + c]-lang_p->syn0[para_index + c])*(lang_p->syn0[index + c]-lang_p->syn0[para_index + c]);
	y*=weight_para;
    if(lang_p->src_tar==0)
    	c1_src_j3+=y;
    else c1_tar_j3+=y;
}

real get_score(long long src_index,long long tar_index,int count)
{
	real y1=0,y2=0,y=0;
	int src_trans,tar_trans;
	int c;
	src_trans=src_lang->vocab[src_index].trans_index;
	tar_trans=tar_lang->vocab[tar_index].trans_index;
    for(c=0;c<src_lang->trans[src_trans].num_trans;c++) 
    	if(tar_trans==src_lang->trans[src_trans].trans_indexs[c])
    	{
    		y1=src_lang->trans[src_trans].trans_pro[c];
    		break;
    	}
    for(c=0;c<tar_lang->trans[tar_trans].num_trans;c++) 
    	if(src_trans==tar_lang->trans[tar_trans].trans_indexs[c])
    	{
    		y1=tar_lang->trans[tar_trans].trans_pro[c];
    		break;
    	}
    if(y1!=0 && y2!=0) y=-count*(get_log(y1)+get_log(y2));
    return y;

}

// 训练一个Sim Mono 类型2 的实例
void TrainSimMonoType2(real *src_neu1e, real *tar_neu1e, unsigned long long *next_random_p) {
	long long c = 0, d = 0;						// 用于遍历的变量
	long long bi_cooc_table_index = 0;			// 翻译计数表索引
	long long src_index = 0, tar_index = 0;		// 源端词索引 目标端词索引
	long long l1, l2 = 0;						// l1 用于遍历单个src词向量每一维的索引
	long long target = 0;
	int label = 0;
	real y=0;
	int count;
	// l2 用于遍历单个tar词向量每一维的索引

	(*next_random_p) = (*next_random_p) * (unsigned long long)25214903917 + 11;
	bi_cooc_table_index = bi_cooc_sample_table[((*next_random_p) >> 16) % table_size];

	src_index = bi_cooc_table[bi_cooc_table_index].src_index;
	tar_index = bi_cooc_table[bi_cooc_table_index].tar_index;
	count=bi_cooc_table[bi_cooc_table_index].count;
	y=get_score(src_index,tar_index,count);
	c2_j3+=y;
	if (src_index == -1 || tar_index == -1 || bi_cooc_table[bi_cooc_table_index].count < BI_COOC_THRESHOLD)
		return;
    for (c = 0; c < layer1_size; c++) src_neu1e[c] = tar_neu1e[c] = 0;
	// 源语言
	l1 = src_index * layer1_size; 
	if (negative > 0) for (d = 0; d < negative + 1; d++) {
		if (d == 0) {
			target = tar_index;
			label = 1;
		}
		else {
			(*next_random_p) = (*next_random_p) * (unsigned long long)25214903917 + 11;
			target = tar_lang->sample_table[((*next_random_p) >> 16) % table_size];
			if (target == 0) target = (*next_random_p) % (tar_lang->vocab_size - 1) + 1;
			if (target == tar_index) continue;
			label = 0;
		}
		l2 = target * layer1_size;
		NegativeSample((src_lang->syn0) + l1, (tar_lang->syn1neg) + l2, src_neu1e, label,weight_mono);
	}
	// 目标语言
	l2 = tar_index * layer1_size;
	if (negative > 0) for (d = 0; d < negative + 1; d++) {
		if (d == 0) {
			target = src_index;
			label = 1;
		}
		else {
			(*next_random_p) = (*next_random_p) * (unsigned long long)25214903917 + 11;
			target = src_lang->sample_table[((*next_random_p) >> 16) % table_size];
			if (target == 0) target = (*next_random_p) % (src_lang->vocab_size - 1) + 1;
			if (target == src_index) continue;
			label = 0;
		}
		l1 = target * layer1_size;
		NegativeSample((tar_lang->syn0) +l2, (src_lang->syn1neg) + l1, tar_neu1e, label,weight_mono);
	}
	l1 = src_index * layer1_size;
	l2 = tar_index * layer1_size;
	for (c = 0; c < layer1_size; c++) src_lang->syn0[l1 + c] += src_neu1e[c];
	for (c = 0; c < layer1_size; c++) tar_lang->syn0[l2 + c] += tar_neu1e[c];
}

long long max(long long a,long long b)
{
	if(a>b) return a;
	return b;
}

/*
 * src_before		保存源语言原向量	real数组 减少函数内重复申请内存
 * tar_before		保存目标语言原向量	real数组 减少函数内重复申请内存
 * src_neu1e        源语言误差累加		real数组 减少函数内重复申请内存
					src_neu1e = sum(phi(src, tar_k') * Wk')
 * tar_neu1e        目标语言误差累加	real数组 减少函数内重复申请内存
					tar_neu1e = sum(phi(src_k, tar) * Wk)
 * next_random_p	随机数指针			整型指针 为了与调用函数同步
 */
// SRC TAR
/*
void TrainSimMonoType2(real *src_before, real *tar_before, real *src_neu1e, real *tar_neu1e, 
					   unsigned long long *next_random_p) {
	// loss  = -(loss1 + loss2)
	// loss1 = log P(w_src | w_tar) 
	// loss2 = log P(w_tar | w_src)
	long long c = 0, d = 0;						// 用于遍历的变量
	long long bi_cooc_table_index = 0;			// 翻译计数表索引
	long long src_index = 0, tar_index = 0;		// 源端词索引 目标端词索引
	long long src_trans_index = 0, tar_trans_index = 0;	// 源端词翻译个数 目标端词翻译个数
	long long l1, l2 = 0;						// l1 用于遍历单个src词向量每一维的索引
												// l2 用于遍历单个tar词向量每一维的索引

	// phi_src_k	用于存储	phi(src_k, tar)		phi(src, tar)不存储在phi_src_k中
	// phi_tar_k	用于存储	phi(src, tar_k')	phi(src, tar)不存储在phi_tar_k中
	// phi_src_tar	用于存储	phi(src, tar)
	real *phi_src_k = NULL, *phi_tar_k = NULL, phi_src_tar = 0;

	// sum_phi_src	phi_src_tar + sum(phi_src_k)
	// sum_phi_tar	phi_src_tar + sum(phi_tar_k)
	real sum_phi_src = 0, sum_phi_tar = 0, g = 0;

	(*next_random_p) = (*next_random_p) * (unsigned long long)25214903917 + 11;
	bi_cooc_table_index = bi_cooc_sample_table[((*next_random_p) >> 16) % table_size];

	src_index = bi_cooc_table[bi_cooc_table_index].src_index;
	tar_index = bi_cooc_table[bi_cooc_table_index].tar_index;
	src_trans_index = src_lang->vocab[src_index].trans_index;
	tar_trans_index = tar_lang->vocab[tar_index].trans_index;
	if (src_index == -1 || tar_index == -1 || src_trans_index == -1 || tar_trans_index == -1 || bi_cooc_table[bi_cooc_table_index].count < BI_COOC_THRESHOLD)
		return;

	phi_src_k = (real *)calloc(tar_lang->trans[tar_trans_index].num_trans, sizeof(real));
	phi_tar_k = (real *)calloc(src_lang->trans[src_trans_index].num_trans, sizeof(real));
	// 将源语言、目标语言原向量 缓存到src_before tar_before中
	for (c = 0; c < layer1_size; c++) src_before[c] = src_lang->syn0[src_index * layer1_size + c];
	for (c = 0; c < layer1_size; c++) tar_before[c] = tar_lang->syn0[tar_index * layer1_size + c];
	for (c = 0; c < layer1_size; c++) src_neu1e[c] = tar_neu1e[c] = 0;

	// TODO 添加亚采样
	// 源语言计算
	for (c = 0; c < src_lang->trans[src_trans_index].num_trans; c++) {
		if (src_lang->trans[src_trans_index].trans_indexs[c] == tar_index) {
			// 所翻译当前词 计算phi_src_tar
			// phi(src_index, tar_index)
			l1 = src_index * layer1_size;
			l2 = tar_index * layer1_size;
			for (d = 0; d < layer1_size; d++)
				phi_src_tar += src_lang->syn0[l1 + d] * tar_lang->syn0[l2 + d];
			phi_src_tar = exp(phi_src_tar);
			sum_phi_src += phi_src_tar;
			sum_phi_tar += phi_src_tar;
		}
		else { 
			// 所翻译的其他词 计算phi_tar_k
			// phi(src_index, tar_index_k)
			l1 = src_index * layer1_size;
			l2 = src_lang->trans[src_trans_index].trans_indexs[c] * layer1_size;
			for (d = 0; d < layer1_size; d++)
				phi_tar_k[c] += src_lang->syn0[l1 + d] * tar_lang->syn0[l2 + d];
			phi_tar_k[c] = exp(phi_tar_k[c]);
			sum_phi_tar += phi_tar_k[c];
			for (d = 0; d < layer1_size; d++)
				src_neu1e[d] += (tar_lang->syn0[l2 + d] * phi_tar_k[c]);
		}
	}
	for (c = 0; c < tar_lang->trans[tar_trans_index].num_trans; c++) {
		// 所翻译当前词 计算phi_src_tar
		// phi(src_index, tar_index) 直接跳过
		if (tar_lang->trans[tar_trans_index].trans_indexs[c] == src_index)  continue;// 所翻译当前词
		else { 
			// 所翻译的其他词 计算phi_src_k
			// phi(src_index_k, tar_index)
			l1 = tar_lang->trans[tar_trans_index].trans_indexs[c] * layer1_size;
			l2 = tar_index * layer1_size;
			for (d = 0; d < layer1_size; d++)
				phi_src_k[c] += tar_lang->syn0[l2 + d] * src_lang->syn0[l1 + d];
			phi_src_k[c] = exp(phi_src_k[c]);
			sum_phi_src += phi_src_k[c];
			for (d = 0; d < layer1_size; d++)
				tar_neu1e[d] += src_lang->syn0[l1 + d] * phi_src_k[c];
		}
	}
	// loss = - (loss1 + loss2)
	// 故所有的更新直接取 +=
	// 更新当前翻译词
	l1 = src_index * layer1_size;
	l2 = tar_index * layer1_size;
	g = 2 - phi_src_tar / sum_phi_tar - phi_src_tar / sum_phi_src;
	for (c = 0; c < layer1_size; c++)
		src_lang->syn0[l1 + c] += (g * tar_before[c] + src_neu1e[c] / sum_phi_tar) * alpha;
	for (c = 0; c < layer1_size; c++)
		tar_lang->syn0[l2 + c] += (g * src_before[c] + tar_neu1e[c] / sum_phi_src) * alpha;
	// 更新非当前翻译词
	// 更新目标端词向量
	for (c = 0; c < src_lang->trans[src_trans_index].num_trans; c++) 
	if (src_lang->trans[src_trans_index].trans_indexs[c] != tar_index) {
		g = (1 - phi_tar_k[c] / sum_phi_tar) * alpha;
		l2 = src_lang->trans[src_trans_index].trans_indexs[c] * layer1_size;
		for (d = 0; d < layer1_size; d++) tar_lang->syn0[l2 + d] += g * src_before[d];
	}
	// 更新源端词向量
	for (c = 0; c < tar_lang->trans[tar_trans_index].num_trans; c++)
	if (tar_lang->trans[tar_trans_index].trans_indexs[c] != src_index) {
		g = (1 - phi_src_k[c] / sum_phi_src) * alpha;
		l1 = tar_lang->trans[tar_trans_index].trans_indexs[c] * layer1_size;
		for (d = 0; d < layer1_size; d++) src_lang->syn0[l1 + d] += g * tar_before[d];
	}
	free(phi_src_k);
	free(phi_tar_k);
}*/

// 训练线程函数
// SRC TAR
void *TrainModelThread(void *id) {
	long long word_count_actual = 0, train_words = 0;
	long long src_word_count = 0, src_last_word_count = 0, src_sen[MAX_SENTENCE_LENGTH + 1];
	long long tar_word_count = 0, tar_last_word_count = 0, tar_sen[MAX_SENTENCE_LENGTH + 1];
	long long src_sen_len = 0, src_sen_pos = 0, tar_sen_len = 0, tar_sen_pos = 0;
	long long local_iter = iter;
	unsigned long long src_next_random = (long long)id;
	unsigned long long tar_next_random = (long long)id;
	int src_finish = 1 - src_cooc;
	int tar_finish = 1 - tar_cooc;
	clock_t now;
	FILE *src_fi = NULL, *tar_fi = NULL;
	// 用于缓存的残差
	real *neu1e = (real *)calloc(layer1_size, sizeof(real));
	real *neu2e = (real *)calloc(layer1_size, sizeof(real));
	//real *neu3e = (real *)calloc(layer1_size, sizeof(real));
	//real *neu4e = (real *)calloc(layer1_size, sizeof(real));

	if (src_cooc) {
		src_fi = fopen(src_lang->train_file, "rb");
		fseek(src_fi, src_lang->file_size / (long long)num_threads * (long long)id, SEEK_SET);
	}
	if (tar_cooc) {
		tar_fi = fopen(tar_lang->train_file, "rb");
		fseek(tar_fi, tar_lang->file_size / (long long)num_threads * (long long)id, SEEK_SET);
	}
	
	while (1) {
		// 打印进度 更改学习率
		if ((src_word_count + tar_word_count - src_last_word_count - tar_last_word_count) > 20000) {
			src_lang->word_count_actual += src_word_count - src_last_word_count;
			tar_lang->word_count_actual += tar_word_count - tar_last_word_count;
			src_last_word_count = src_word_count;
			tar_last_word_count = tar_word_count;

			word_count_actual = src_lang->word_count_actual + tar_lang->word_count_actual;
			train_words = src_lang->train_words + tar_lang->train_words;
			if ((debug_mode > 1)) {
				now = clock();
				printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
					word_count_actual / (real)(iter * train_words + 1) * 100,
					word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
				fflush(stdout);
			}
			alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
			if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
		}

		// 读取一个句子
		if (!src_finish)
			ReadSentence(src_lang, src_sen, &src_sen_pos, &src_sen_len, src_fi, &src_word_count, &src_next_random);
		if (!tar_finish)
			ReadSentence(tar_lang, tar_sen, &tar_sen_pos, &tar_sen_len, tar_fi, &tar_word_count, &tar_next_random);
		
		// 判断源语言是否完成了一个迭代
		if (feof(src_fi) || (src_word_count > src_lang->train_words / num_threads)) {
			src_lang->word_count_actual += src_word_count - src_last_word_count;
			src_finish = 1;
			src_word_count = 0;
			src_last_word_count = 0;
			src_sen_len = 0;
			fseek(src_fi, src_lang->file_size / (long long)num_threads * (long long)id, SEEK_SET);
		}

		// 判断目标语言是否完成了一个迭代
		if (feof(tar_fi) || (tar_word_count > tar_lang->train_words / num_threads)) {
			tar_lang->word_count_actual += tar_word_count - tar_last_word_count;
			tar_finish = 1;
			tar_word_count = 0;
			tar_last_word_count = 0;
			tar_sen_len = 0;
			fseek(tar_fi, tar_lang->file_size / (long long)num_threads * (long long)id, SEEK_SET);
		}

		if (src_finish && tar_finish) {
			local_iter--;
			if (local_iter == 0) break;
			src_finish = 1 - src_cooc;
			tar_finish = 1 - tar_cooc;
			continue;
		}

		if (!src_finish)
		{
			TrainOneSGNS(src_lang, src_sen, &src_sen_pos, &src_sen_len, &src_next_random, neu1e);
			//j/=src_lang->vocab_size;
			//printf("Now the scores:%f\n",j);
		}
		if (!tar_finish)
		{
			TrainOneSGNS(tar_lang, tar_sen, &tar_sen_pos, &tar_sen_len, &tar_next_random, neu1e);
			//j/=tar_lang->vocab_size;
			//printf("Now the scores:%f\n",j);
		}
		
		if (bi_sim) {
            TrainSimBi(src_lang, tar_lang, neu1e, &src_next_random);
            //j2/=src_lang->trans_size;
            //j+=j2;
            //printf("Now the scores:%f\n",j);
            TrainSimBi(tar_lang, src_lang, neu1e, &tar_next_random);
            //j2/=tar_lang->trans_size;
            //j+=j2;
            //printf("Now the scores:%f\n",j);
        }

        if (mono_sim == 1) {
            if (!src_finish)
            {
                TrainSimMonoType1(src_lang, neu1e, &src_next_random);
                //printf("Now the scores:%f\n",j);
            }
            if (!tar_finish)
            {
                TrainSimMonoType1(tar_lang, neu1e, &tar_next_random);
                //printf("Now the scores:%f\n",j);
            }
        }
        if (mono_sim == 2) {
            if (!src_finish && !tar_finish)
            {
                TrainSimMonoType2(neu1e, neu2e, &src_next_random);
                //printf("Now the scores:%f\n",j);
            }
        }
	}
	//j2/=max(src_lang->trans_size,tar_lang->trans_size);
	//j+=j2;
	//printf("Final score :%f\n", j);
	/*
	if (!src_fi)      //???
		fclose(src_fi);
	if (!tar_fi)
		fclose(tar_fi);*/
	if(src_fi!=NULL)
		fclose(src_fi);
	if(tar_fi!=NULL)
		fclose(tar_fi);
	free(neu1e);
	free(neu2e);
	//free(neu3e);
	//free(neu4e);
	pthread_exit(NULL);
}

void TrainModel() {
	long long a;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	// printf("Starting training using file %s\n", src_lang->train_file);
	starting_alpha = alpha;
	
    if (src_lang->output_file[0] == 0) { printf("Lack Src Output File Path\n"); exit(1); }
    if (tar_lang->output_file[0] == 0) { printf("Lack Tar Output File Path\n"); exit(1); }
    
	// 准备源语言词表
	if (src_lang->read_vocab_file[0] != 0) ReadVocab(src_lang); else LearnVocabFromTrainFile(src_lang);
	if (src_lang->save_vocab_file[0] != 0) SaveVocab(src_lang);
	if (src_lang->output_file[0] == 0) return;
	InitNet(src_lang);
	if (src_lang->pre_file[0] != 0) LoadPreTrain(src_lang);
	if (negative > 0) InitUnigramTable(src_lang);

	// 准备目标语言词表
	if (tar_lang->read_vocab_file[0] != 0) ReadVocab(tar_lang); else LearnVocabFromTrainFile(tar_lang);
	if (tar_lang->save_vocab_file[0] != 0) SaveVocab(tar_lang);
	if (tar_lang->output_file[0] == 0) return;
	InitNet(tar_lang);
	if (tar_lang->pre_file[0] != 0) LoadPreTrain(tar_lang);
	if (negative > 0) InitUnigramTable(tar_lang);

	// 准备翻译计数表
    if (bi_sim || mono_sim == 2) {
        if (bi_cooc_file[0] != 0) {
            //ReadTransCount(bi_cooc_file);
            ReadTransCount();
            // 初始化翻译技术随机采样表
            InitBicoocSampleTable();
            InitTransPro(src_lang);
            InitTransPro(tar_lang);
        }
        else {
            printf("Lack Bi-Lingual Count File\n"); exit(1);
        }
    }

    // 准备源语言复述表
    if (mono_sim == 1) {
	   if (src_lang->para_file[0] != 0) ReadParaPhrase(src_lang); else {printf("Lack Src Language Paraphrase File\n"); exit(1);}
	   if (tar_lang->para_file[0] != 0) ReadParaPhrase(tar_lang); else {printf("Lack Tar Language Paraphrase File\n"); exit(1);}
    }


	// 训练
	start = clock();
	RecalcWeight();
    printf("Training Objective:\n");
    printf("\tPart Name: YN  Value\n");
    printf("\tMono Cooc: %d   %.2f\n", src_cooc && tar_cooc, weight_cooc);
    printf("\tBili Sim : %d   %.2f\n", bi_sim, weight_bi);
    printf("\tMono Sim : %d   %.2f\n", mono_sim, weight_mono);
	printf("Start Training\n");
	for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

	src_j1/=src_lang->vocab_size;
    tar_j1/=tar_lang->vocab_size;
    src_j2/=src_lang->trans_size;
    tar_j2/=tar_lang->trans_size;
    if(mono_sim==1)
    {
    	c1_src_j3/=src_lang->para_size;
    	c1_tar_j3/=tar_lang->para_size;
    }
    else if(mono_sim==2)
    	c2_j3/=max(src_lang->trans_size,tar_lang->trans_size);


    printf("Mono Cooc Score:%f    %f\n", src_j1,tar_j1);
    printf("Bili Sim: %f    %f\n", src_j2,tar_j2);
    if(mono_sim==1)
    	printf("Mono Sim:%f   %f\n", c1_src_j3,c1_tar_j3);
    else if(mono_sim==2)
    	printf("Mono Sim:%f\n", c2_j3);

    j= (src_j1+tar_j1)*weight_cooc + (src_j2+tar_j2)* weight_bi;
    if(mono_sim==1)
    	j+=(c1_src_j3+c1_tar_j3)*weight_mono;
    else if(mono_sim==2)
    	j+=c2_j3*weight_mono;
    j+=0.5*(weight_cooc*weight_cooc + weight_bi*weight_bi + weight_mono*weight_mono);
	printf("Final score :%f\n", j);

	// Save the word vectors
	SaveVectors(src_lang);
	SaveVectors(tar_lang);
	
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {//如果为最后一个的话，就没有响应的参数，参数名后面跟相应的参数
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	if (argc == 1) {
		printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-src-train <file>\n");
		printf("\t\tUse text data from <file> to train the src model\n");
		printf("\t-src-output <file>\n");
		printf("\t\tUse <file> to save the resulting the src word vectors / word clusters\n");
		printf("\t-src-para <file>\n");
		printf("\t\tUse src paraphrase rules from <file>\n");
		printf("\t-tar-train <file>\n");
		printf("\t\tUse text data from <file> to train the tar model\n");
		printf("\t-tar-output <file>\n");
		printf("\t\tUse <file> to save the resulting the tar word vectors / word clusters\n");
		printf("\t-tar-para <file>\n");
		printf("\t\tUse tar paraphrase rules from <file>\n");
		printf("\t-bi-cooc <file>\n");
		printf("\t\tUse src and tar co-occurrence from <file>\n");
		printf("\t-size <int>\n");
		printf("\t\tSet size of word vectors; default is 100\n");
		printf("\t-window <int>\n");
		printf("\t\tSet max skip length between words; default is 5\n");
		printf("\t-sample <float>\n");
		printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
		printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 12)\n");
		printf("\t-iter <int>\n");
		printf("\t\tRun more training iterations (default 5)\n");
		printf("\t-min-count <int>\n");
		printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
		printf("\t-alpha <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
		printf("\t-debug <int>\n");
		printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
		printf("\t-src-save-vocab <file>\n");
		printf("\t\tThe src vocabulary will be saved to <file>\n");
		printf("\t-src-read-vocab <file>\n");
		printf("\t\tThe src vocabulary will be read from <file>, not constructed from the training data\n");
		printf("\t-tar-save-vocab <file>\n");
		printf("\t\tThe tar vocabulary will be saved to <file>\n");
		printf("\t-tar-read-vocab <file>\n");
		printf("\t\tThe tar vocabulary will be read from <file>, not constructed from the training data\n");
		printf("\t-bi-sim <int>\n");
		printf("\t\tUse the bi-lingual sim model; default is 1 (use 0 for not use)\n");
        printf("\t-mono-sim <int>\n");
		printf("\t\tUse the mono sim model; default is 0 (use 1 for paraphrase mono sim; use 2 for tranlation mono sim)\n");
		printf("\nExamples:\n");
		printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
		return 0;
	}

	src_lang = (language_p)calloc(1, sizeof(language));
	tar_lang = (language_p)calloc(1, sizeof(language));
	init_language(src_lang, SRC);
	init_language(tar_lang, TAR);

	if ((i = ArgPos((char *)"-src-train", argc, argv)) > 0) strcpy(src_lang->train_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-src-save-vocab", argc, argv)) > 0) strcpy(src_lang->save_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-src-read-vocab", argc, argv)) > 0) strcpy(src_lang->read_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-src-output", argc, argv)) > 0) strcpy(src_lang->output_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-src-para", argc, argv)) > 0) strcpy(src_lang->para_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-src-pre", argc, argv)) > 0) strcpy(src_lang->pre_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-tar-train", argc, argv)) > 0) strcpy(tar_lang->train_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-tar-save-vocab", argc, argv)) > 0) strcpy(tar_lang->save_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-tar-read-vocab", argc, argv)) > 0) strcpy(tar_lang->read_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-tar-output", argc, argv)) > 0) strcpy(tar_lang->output_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-tar-para", argc, argv)) > 0) strcpy(tar_lang->para_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-tar-pre", argc, argv)) > 0) strcpy(tar_lang->pre_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-bi-cooc", argc, argv)) > 0) strcpy(bi_cooc_file, argv[i + 1]);

	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-bi-sim", argc, argv)) > 0) bi_sim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-mono-sim", argc, argv)) > 0) mono_sim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-weight-cooc", argc, argv)) > 0) weight_cooc = (real)atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-weight-bi", argc, argv)) > 0) weight_bi = (real)atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-weight-mono", argc, argv)) > 0) weight_mono = (real)atof(argv[i + 1]);
    
	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	for (i = 0; i < EXP_TABLE_SIZE; i++) {
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}
	TrainModel();
	return 0;
}
