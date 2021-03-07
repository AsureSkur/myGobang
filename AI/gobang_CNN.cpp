#include <iostream>
#include <cstdio>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2\imgproc\types_c.h>

#define TESTNUM 10
#define MAXDEPTH 4
#define MOVENUM 10 
#define WINFLAG 100000
#define BLACK 1
#define WHITE 2
#define BORDER 3
#define LAYERSIZE 17
#define FILTERSIZE 5
#define LEARNINGRATE 0.01
#define TRAINTIMES 5

struct Point2d
{
	int x;
	int y;
	int val;
};
struct Line2d
{
	Point2d st;
	int pos; //1 for (1,0); 2 for (0,1); 3 for (1,1); 4 for (1,-1)
	int length;
	int side;
};
struct Board
{
	int grid[17][17];
};

//use cnn for evaluating state
class layer//define layer
{
public:
	int length, width, height;
	double m[LAYERSIZE][LAYERSIZE][5];
	double b[LAYERSIZE][LAYERSIZE][5];
	double delta[LAYERSIZE][LAYERSIZE][5];
	layer()
	{
		length = 0;
		width = 0;
		height = 0;
		memset(m, 0, sizeof(m));
		memset(b, 0, sizeof(b));
		memset(delta, 0, sizeof(delta));

	}
	layer(int len, int wid, int h)
	{
		length = len;
		width = wid;
		height = h;
		memset(m, 0, sizeof(m));
		memset(b, 0, sizeof(b));
		memset(delta, 0, sizeof(delta));
	}
	layer(layer& ipt)
	{
		length = ipt.length;
		width = ipt.width;
		height = ipt.height;
		for (int i = 0; i < LAYERSIZE; i++)
		{
			for (int j = 0; j < LAYERSIZE; j++)
			{
				for (int k = 0; k < 5; k++)
				{
					m[i][j][k] = ipt.m[i][j][k];
					b[i][j][k] = ipt.b[i][j][k];
					delta[i][j][k] = ipt.delta[i][j][k];
				}
			}
		}
					
	}
	~layer() {}
	void randVal()
	{
		for (int i = 0; i < length; i++)
		{
			for (int j = 0; j < width; j++)
			{
				for (int k = 0; k < height; k++)
				{
					m[i][j][k] = 0.01 * (rand() % 100);
				}
			}
		}
	}
	void chessboardInput(struct Board& board,int role);
	void conv(layer* filter,layer& dst, int number);
	void delta_from_pool_to_conv(layer& pool_layer);
	void filter_update(layer&input_layer,layer& conv_layer,int serial);
	void pooling_max(layer& conv_layer);
};

class fc_layer
{
public:
	int length;
	double m[1000];
	double b[1000];
	double delta[1000];
	double w[20][1000];
	fc_layer()
	{
		length = 0;
		memset(m, 0, sizeof(m));
		memset(b, 0, sizeof(b));
		memset(delta, 0, sizeof(delta));
		for (int i = 0; i < 20; i++)
		{
			for (int j = 0; j < 1000; j++)
			{
				w[i][j] = 0.01 * (rand() % 100);
			}
		}
	}
	void pool_to_fc(layer& conv_layer);
	void softmax();
	void fc_forward(fc_layer& fc_w, fc_layer& fc_dst);
};

class CNNNetwork
{
public:
	layer input_layer;
	layer conv_layer;
	layer pool_layer;
	layer filter[5];
	fc_layer fc_input;
	fc_layer fc_w;
	fc_layer fc_out;
	CNNNetwork()
	{
		for (int i = 0; i < 5; i++)
		{
			filter[i].length = FILTERSIZE;
			filter[i].width = FILTERSIZE;
			filter[i].height = 1;
			filter[i].randVal();
		}
	}
	void delta_from_fc_to_pool(fc_layer& fc, layer& pool_layer);
	//0 for self, 1 for enemy, value is self-enemy * weight; when lose, winflag=2, else winflag=1
	int forward_backward(struct Board chessboard,int winflag,int role)
	{
		double res;
		input_layer.chessboardInput(chessboard,role);
		input_layer.conv(filter, conv_layer, 5);
		pool_layer.pooling_max(conv_layer);
		fc_input.pool_to_fc(pool_layer);
		fc_input.fc_forward(fc_w, fc_out);
		fc_out.softmax();
		if (winflag)
		{
			if (winflag == 1)
			{
				fc_out.delta[0] = fc_out.m[0] - 1.0;
				fc_out.delta[1] = fc_out.m[1];
			}
			else
			{
				fc_out.delta[1] = fc_out.m[1] - 1.0;
				fc_out.delta[0] = fc_out.m[0];
			}
			memset(fc_input.delta, 0, sizeof(fc_input.delta));
			for (int i = 0; i < fc_input.length; i++)
			{
				for (int j = 0; j < 2; j++)
				{
					fc_input.delta[i]+= fc_input.m[i] * (1.0 - fc_input.m[i]) * fc_w.w[j][i] * fc_out.delta[j];
				}
			}
			for (int i = 0; i < fc_input.length; i++)
			{
				for (int j = 0; j < 2; j++)
				{
					fc_w.w[j][i] -= LEARNINGRATE * fc_out.delta[j] * fc_input.m[i];
					fc_w.b[j] -= LEARNINGRATE * fc_out.delta[j];
				}
			}
			delta_from_fc_to_pool(fc_input,pool_layer);
			conv_layer.delta_from_pool_to_conv(pool_layer);
			for (int i = 0; i < FILTERSIZE; i++)
			{
				filter[i].filter_update(input_layer, conv_layer, i);
			}
		}
		res = (fc_out.m[0] - fc_out.m[1]) * 10000;
		return (int)res;
	}

};

double Relu(double x);
double sigmod(double x);
int findfive(struct Board& board);
int stateEvaluate(struct Board& board, int maxColor, CNNNetwork& cnn, int winflag = 0);
void showBoard(struct Board& board);
void initBoard(struct Board& board);
int setBoard(char* filepath,struct Board& board);
//alpha-beta cut pruning search func
int alphabetaSearch(int depth, int alpha, int beta, bool role, int color, struct Board& board, CNNNetwork& cnn,int& winflag);
void generateMove(std::vector<struct Point2d>& points, struct Board& board);

CNNNetwork cnn;
struct Board board;

int main(int argc, const char* argv[])
{
	initBoard(board);
	char filepath[260];
	int color, winflag,value;
	//train
	printf("Begin training!\n");
	for (int t = 0; t < TRAINTIMES; t++)
	{
		printf("Times:%d\n", t);
		for (int i = 1; i <= TESTNUM; i++)
		{
			initBoard(board);
			char tmp[20];
			memset(filepath, 0, sizeof(filepath));
			itoa(i, tmp, 10);
			strcpy(filepath, "data/label");
			strcat(filepath, tmp);
			strcat(filepath, ".txt");
			//printf("%s\n", filepath);
			color = setBoard(filepath, board);
			winflag = 0;
			value = alphabetaSearch(MAXDEPTH, INT_MIN, INT_MAX, 1, color, board, cnn, winflag);
			printf("Final value of map:%d is %d\n", i, value);
		}
		printf("Times:%d End!\n\n", t);
	}
	printf("Succeed!\n");

	//test
	initBoard(board);
	printf("Input chess status filename. if none input NONE\n");
	memset(filepath, 0, sizeof(filepath));
	scanf("%s", filepath);
	getchar();
	color = 1;
	if (strcmp(filepath, "NONE") != 0)
	{
		color = setBoard(filepath, board);
	}
	winflag = 0;
	while (1)
	{
		value = stateEvaluate(board, (color == 1) ? BLACK : WHITE, cnn);
		if (value >= WINFLAG || value <= -WINFLAG)
		{
			break;
		}
		std::vector<struct Point2d>bestmoves;
		generateMove(bestmoves, board);
		for (int i = 0; i < bestmoves.size(); i++)
		{
			int x = bestmoves[i].x;
			int y = bestmoves[i].y;
			board.grid[x][y] = (color == 1) ? BLACK : WHITE;
			bestmoves[i].val = alphabetaSearch(MAXDEPTH, INT_MIN, INT_MAX, 1, color, board, cnn, winflag);
			board.grid[x][y] = 0;
		}
		int pos = 0;
		for (int i = 0; i < bestmoves.size(); i++)
		{
			if (bestmoves[i].val > bestmoves[pos].val)
			{
				pos = i;
			}
		}
		int x = bestmoves[pos].x, y = bestmoves[pos].y;
		printf("AI placed chess on ( %d, %d )\n", x, y);
		board.grid[x][y] = (color == 1) ? BLACK : WHITE;
		showBoard(board);
		value = stateEvaluate(board, (color == 1) ? BLACK : WHITE, cnn);
		if (value >= WINFLAG || value <= -WINFLAG)
		{
			break;
		}
		while (1)
		{
			x = 0;
			y = 0;
			printf("What is your move? Input two number.\n");
			scanf("%d %d", &x, &y);
			getchar();
			if (board.grid[x][y] == 0)
			{
				board.grid[x][y] = (color != 1) ? BLACK : WHITE;
				break;
			}
			else
			{
				printf("Input place has been set a chess!\n");
			}
		}

	}

	return 0;
}

int setBoard(char* filepath,struct Board& board)
{
	int color;
	FILE* fp = freopen(filepath, "r", stdin);
	int x, y, type;
	int black = 0, white = 0;
	while (scanf("%d %d %d", &x, &y, &type) != EOF)
	{
		board.grid[x + 1][y + 1] = (type == 1) ? BLACK : WHITE;
		if (type == 1)
		{
			black++;
		}
		else white++;
	}
	freopen("CON", "r", stdin);
	color = (black > white) ? -1 : 1;
	return color;
}

void initBoard(struct Board& board)
{
	for (int i = 0; i < 17; i++)
	{
		for (int j = 0; j < 17; j++)
		{
			if (i == 0 || j == 0 || i == 16 || j == 16)
			{
				board.grid[i][j] = BORDER;
			}
			else board.grid[i][j] = 0;
		}
	}
}

void showBoard(struct Board& board)
{
	printf("Chessboard:\n");
	for (int i = 1; i < 16; i++)
	{
		for (int j = 1; j < 16; j++)
		{
			printf(" %d", board.grid[i][j]);
		}
		printf("\n");
	}
	printf("\n");
	return;
}

int stateEvaluate(struct Board& board, int maxColor, CNNNetwork& cnn, int winflag)
{
	int color = findfive(board);
	if (color)
	{
		if (color == maxColor)
		{
			winflag = 1;
		}
		else
		{
			winflag = 2;
		}
	}
	int value = cnn.forward_backward(board, winflag, maxColor);

	if (!color)
	{
		return value;
	}
	else if(color == 1)
	{
		return WINFLAG + 1;
	}
	else
	{
		return -WINFLAG - 1;
	}

}

//start from ai who is the "MAX" player as the other is "MIN" player
int alphabetaSearch(int depth, int alpha, int beta, bool role, int color, struct Board& board,CNNNetwork& cnn,int& winflag)
{
	int maxColor;
	if (role)
	{
		maxColor = (color == 1) ? BLACK : WHITE;
	}
	else
	{
		maxColor = (color == -1) ? BLACK : WHITE;
	}
	int value = 0;
	value = stateEvaluate(board, maxColor, cnn, winflag);
	if (depth == 0 || value <= -WINFLAG || value >= WINFLAG)
	{
		if (value <= -WINFLAG)
		{
			winflag = 2;
		}
		if (value >= WINFLAG)
		{
			winflag = 1;
		}
		return value;
	}
	std::vector<struct Point2d>moves;
	generateMove(moves, board);
	for (int i = 0; i < moves.size(); i++)
	{
		int x = moves[i].x;
		int y = moves[i].y;
		board.grid[x][y] = (color == 1) ? BLACK : WHITE;
		int tmp = alphabetaSearch(depth - 1, alpha, beta, !role, -color, board, cnn, winflag);
		value = stateEvaluate(board, maxColor, cnn, winflag);
		moves[i].val = winflag;
		board.grid[x][y] = 0;
		winflag = 0;
		if (role)//ai is 1; the other is 0
		{
			if (alpha < tmp)
			{
				alpha = tmp;
			}
			if (alpha >= beta)
			{
				break;
			}
		}
		else
		{
			if (beta > tmp)
			{
				beta = tmp;
			}
			if (alpha >= beta)
			{
				break;
			}
		}
	}
	if (role)
	{
		return alpha;
	}
	else
	{
		return beta;
	}
}

void generateMove(std::vector<struct Point2d>& points, struct Board& board)
{
	Point2d tmp;
	tmp.val = 0;
	struct Board board_tmp;
	initBoard(board_tmp);
	points.clear();
	for (int x = 1; x < 16; x++)
	{
		for (int y = 1; y < 16; y++)
		{
			if (board.grid[x][y])
			{
				if (board.grid[x - 1][y] == 0)
				{
					board_tmp.grid[x - 1][y] = 1;
				}
				if (board.grid[x - 1][y - 1] == 0)
				{
					board_tmp.grid[x - 1][y - 1] = 1;
				}
				if (board.grid[x - 1][y + 1] == 0)
				{
					board_tmp.grid[x - 1][y + 1] = 1;
				}
				if (board.grid[x][y - 1] == 0)
				{
					board_tmp.grid[x][y - 1] = 1;
				}
				if (board.grid[x][y + 1] == 0)
				{
					board_tmp.grid[x][y + 1] = 1;
				}
				if (board.grid[x + 1][y] == 0)
				{
					board_tmp.grid[x + 1][y] = 1;
				}
				if (board.grid[x + 1][y - 1] == 0)
				{
					board_tmp.grid[x + 1][y - 1] = 1;
				}
				if (board.grid[x + 1][y + 1] == 0)
				{
					board_tmp.grid[x + 1][y + 1] = 1;
				}
			}
		}
	}
	for (int i = 1; i < 16; i++)
	{
		tmp.x = i;
		for (int j = 1; j < 16; j++)
		{
			tmp.y = j;
			if (board_tmp.grid[i][j])
			{
				points.push_back(tmp);
			}
		}
	}
	if (points.size() == 0 && board.grid[7][7] == 0)
	{
		tmp.x = 7;
		tmp.y = 7;
		points.push_back(tmp);
	}
	return;
}

int findfive(struct Board& board)
{
	int left,down,slant1,slant2; //judge left, down and slant_down
	int length;
	for (int x = 1; x < 16; x++)
	{
		for (int y = 1; y < 16; y++)
		{
			if (board.grid[x][y] == 0)
			{
				continue;
			}
			//initial a temp line
			length = 1;

			left = 16 - x;
			down = 16 - y;
			slant1 = left < down ? left : down;
			slant2 = ((16 - x) < y) ? (16 - x) : y;

			//reset length = 1 and side = 0
			length = 1;
			if (board.grid[x][y] != board.grid[x - 1][y])
			{
				for (int i = 1; i < left; i++)
				{
					if (board.grid[x][y] == board.grid[x + i][y])
					{
						length += 1;
					}
					else
					{
						break;
					}

				}
			}
			//only push single chess at first time, and reset length and side
			if (length >= 5)
			{
				return board.grid[x][y];
			}
			length = 1;

			if (board.grid[x][y] != board.grid[x][y - 1])
			{
				for (int i = 1; i < down; i++)
				{
					if (board.grid[x][y] == board.grid[x][y + i])
					{
						length += 1;
					}
					else
					{
						break;
					}
				}
			}
			if (length >= 5)
			{
				return board.grid[x][y];
			}
			length = 1;
			if (board.grid[x][y] != board.grid[x - 1][y - 1])
			{
				for (int i = 1; i < slant1; i++)
				{
					if (board.grid[x][y] == board.grid[x + i][y + i])
					{
						length += 1;
					}
					else
					{
						break;
					}
				}
			}
			if (length >= 5)
			{
				return board.grid[x][y];
			}
			length = 1;
			if (board.grid[x][y] != board.grid[x - 1][y + 1])
			{
				for (int i = 1; i < slant2; i++)
				{
					if (board.grid[x][y] == board.grid[x + i][y - i])
					{
						length += 1;
					}
					else
					{
						break;
					}
				}
			}
			if (length >= 5)
			{
				return board.grid[x][y];
			}

		}
	}
	return 0;
}


double Relu(double x)
{
	if (x < 0)
	{
		x = 0;
	}
	return x;
}

double sigmod(double x)
{
	double res = 1.0 / (1.0 + exp(-x));
	return res;
}

void layer::conv(layer* filter,layer& dst, int number)
{
	dst.length = length - FILTERSIZE + 1;
	dst.width = width - FILTERSIZE + 1;
	dst.height = number * height;
	int h, i, j, a, b, k;
	for (h = 0; h < dst.height; h++)
	{
		for (i = 0; i < dst.length; i++)
		{
			for (j = 0; j < dst.width; j++)
			{
				for (a = 0; a < FILTERSIZE; a++)
				{
					for (b = 0; b < FILTERSIZE; b++)
					{
						for (k = 0; k < height; k++)
						{
							dst.m[i][j][h] += m[i + a][j + b][k] * filter[h].m[a][b][0];
						}
					}
				}
				dst.m[i][j][h] = Relu(dst.m[i][j][h] + dst.b[i][j][h]);
			}
		}
	}
	return;
}

void layer::chessboardInput(struct Board& board,int role)
{
	length = 15;
	width = 15;
	height = 1;
	for (int i = 1; i < 16; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			if (board.grid[i][j])
			{
				m[i][j][0] = (board.grid[i][j] == role) ? 1 : -1;
			}
		}
	}
}

void CNNNetwork::delta_from_fc_to_pool(fc_layer& fc,layer& pool_layer)
{
	int x = 1;
	for (int i = 0; i < pool_layer.length; i++)
	{
		for (int j = 0; j < pool_layer.width; j++)
		{
			for (int k = 0; k < pool_layer.height; k++)
			{
				pool_layer.delta[i][j][k] = fc.delta[x];
				x++;
			}
		}
	}
	return;
}

void layer::delta_from_pool_to_conv(layer& pool_layer)
{
	for (int k = 0; k < height; k++)
	{
		for (int i = 0; i < length; i++)
		{
			for (int j = 0; j < width; j++)
			{

				if (fabs(m[i][j][k] - pool_layer.m[i / 2][j / 2][k]) < 0.01) 
				{
					delta[i][j][k] = pool_layer.delta[i / 2][j / 2][k];
				}
				else delta[i][j][k] = 0;
			}
		}
	}
}

void layer::filter_update(layer&input_layer,layer& conv_layer,int serial)
{
	double sum = 0.0;
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < width; j++)
		{
			sum = 0.0;
			for (int x = 0; x < FILTERSIZE; x++)
			{
				for (int y = 0; y < FILTERSIZE; y++)
				{
					sum += conv_layer.delta[i][j][serial] * input_layer.m[i + x][j + y][0];
				}
			}
			m[i][j][0] -= LEARNINGRATE * sum;
		}
	}
	for (int i = 0; i < conv_layer.length; i++)
	{
		for (int j = 0; j < conv_layer.width; j++)
		{
			conv_layer.b[i][j][serial] -= LEARNINGRATE * conv_layer.delta[i][j][serial];
		}
	}
	return;
}

void layer::pooling_max(layer& conv_layer)
{
	length = conv_layer.length / 2;
	width = conv_layer.width / 2;
	height = conv_layer.height;
	for (int k = 0; k < conv_layer.height; k++) 
	{
		for (int i = 0; i < conv_layer.length; i += 2) 
		{
			for (int j = 0; j < conv_layer.width; j += 2) 
			{
				double tmp;
				m[i / 2][j / 2][k] = conv_layer.m[i][j][k];
				tmp = conv_layer.m[i + 1][j][k];
				m[i / 2][j / 2][k] = m[i / 2][j / 2][k] > tmp ? m[i / 2][j / 2][k] : tmp;
				tmp = conv_layer.m[i][j + 1][k];
				m[i / 2][j / 2][k] = m[i / 2][j / 2][k] > tmp ? m[i / 2][j / 2][k] : tmp;
				tmp = conv_layer.m[i + 1][j + 1][k];
				m[i / 2][j / 2][k] = m[i / 2][j / 2][k] > tmp ? m[i / 2][j / 2][k] : tmp;
			}
		}
	}
	return;
}

void fc_layer::pool_to_fc(layer& pool_layer)
{
	m[0] = 1.0;
	int x = 1;
	for (int i = 0; i < pool_layer.length; i++)
	{
		for (int j = 0; j < pool_layer.width; j++)
		{
			for (int k = 0; k < pool_layer.height; k++)
			{
				m[x] = sigmod(pool_layer.m[i][j][k]);
				x++;
			}
		}
	}
	length = x;
}

void fc_layer::softmax() //divided into two side, black win (0) or white win (1)
{
	double sum = 0.0; 
	double max = -INFINITY;
	for (int i = 0; i < 2; i++) 
	{
		max = max > m[i] ? max : m[i];
	}
	for (int i = 0; i < 2; i++) 
	{
		sum += exp(m[i] - max);
	}
	for (int i = 0; i < 2; i++) 
	{
		m[i] = exp(m[i] - max) / sum;
	}
	return;
}

void fc_layer::fc_forward(fc_layer& fc_w, fc_layer& dst)
{
	dst.length = 2;
	for (int i = 0; i < dst.length; i++)
	{
		for (int j = 0; j < length; j++)
		{
			dst.m[i] += fc_w.w[i][j] * m[j];
		}
		dst.m[i] += fc_w.b[i];
	}
	return;
}