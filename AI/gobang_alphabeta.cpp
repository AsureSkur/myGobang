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

void showBoard(struct Board& board);
void initBoard(struct Board& board);

//alpha-beta cut pruning search func
int alphabetaSearch(int depth, int alpha, int beta, bool role, int color, struct Board& board);
void generateMove(std::vector<struct Point2d>& points, struct Board& board);

//handwork evaluating func
void findChesslines(std::vector<struct Line2d>& lines, struct Board& board);
int stateEvaluate(struct Board& board, int maxColor);

int main(int argc, const char* argv[])
{
	struct Board board;
	initBoard(board);
	char filepath[260];
	printf("Input chess status filename. if none input NONE\n");
	scanf("%s", filepath);
	getchar();
	int color = 1;
	if (strcmp(filepath, "NONE") != 0)
	{
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
	}
	int value;
	while (1)
	{
		value = stateEvaluate(board, (color == 1) ? BLACK : WHITE);
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
			bestmoves[i].val = alphabetaSearch(MAXDEPTH, INT_MIN, INT_MAX, 1, color, board);
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
		value = stateEvaluate(board, (color == 1) ? BLACK : WHITE);
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
	if (value >= WINFLAG)
	{
		printf("AI win!\n");
	}
	else
	{
		printf("Player win!\n");
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


//start from ai who is the "MAX" player as the other is "MIN" player
int alphabetaSearch(int depth, int alpha, int beta, bool role, int color, struct Board& board)
{
	int maxColor;
	if (role)
	{
		maxColor = (color == 1) ? BLACK : WHITE;
	}
	else
	{
		maxColor = (color == 0) ? BLACK : WHITE;
	}
	int value = stateEvaluate(board, maxColor);
	if (depth == 0 || value <= -WINFLAG || value >= WINFLAG)
	{
		return value;
	}
	std::vector<struct Point2d>moves;
	generateMove(moves, board);
	for (int i = 0; i < moves.size(); i++)
	{
		int x = moves[i].x;
		int y = moves[i].y;
		board.grid[x][y] = (color == 1) ? BLACK : WHITE;
		int tmp = alphabetaSearch(depth - 1, alpha, beta, !role, -color, board);
		board.grid[x][y] = 0;
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

int stateEvaluate(struct Board& board, int maxColor)
{
	std::vector<struct Line2d> lines;
	findChesslines(lines, board);

	int value = 0;
	//handmake value count method: one as 1, two as 10, three as 100, four as 1000, five as 100000. If been blocked, value lowered a grade
	for (int i = 0; i < lines.size(); i++)
	{
		int tmp = 0;
		if (lines[i].length >= 5)
		{
			value = WINFLAG + 1;
			if (maxColor != lines[i].st.val)
			{
				value = -value;
			}
			return value;
		}
		switch (lines[i].length)
		{
		case 1:tmp = 10; break;
		case 2:tmp = 100; break;
		case 3:tmp = 1000; break;
		case 4:tmp = 10000; break;
		case 5:tmp = WINFLAG * 10; break;
		}
		if (maxColor == lines[i].st.val)
		{
			if (lines[i].side == 0)
			{
				tmp = 0;
			}
			else if (lines[i].side == 1)
			{
				tmp = tmp / 10;
			}
			else if (lines[i].side > 2)
			{
				tmp += lines[i].side;
			}
			value += tmp;
		}
		else
		{
			if (lines[i].side == 0)
			{
				tmp = 0;
			}
			else if (lines[i].side == 1)
			{
				tmp = tmp / 10;
			}
			value -= tmp;
		}
	}

	return value;
}
void findChesslines(std::vector<struct Line2d>& lines, struct Board& board)
{
	int left, down, slant1, slant2; //judge left, down and slant_down
	struct Line2d line;
	for (int x = 1; x < 16; x++)
	{
		for (int y = 1; y < 16; y++)
		{
			if (board.grid[x][y] == 0)
			{
				continue;
			}
			//initial a temp line
			line.st.x = x;
			line.st.y = y;
			line.st.val = board.grid[x][y];
			line.pos = 0;
			line.length = 1;
			line.side = 0;

			left = 16 - x;
			down = 16 - y;
			slant1 = left < down ? left : down;
			slant2 = ((16 - x) < y) ? (16 - x) : y;

			//reset length = 1 and side = 0
			line.length = 1;
			line.side = 0;
			if (board.grid[x - 1][y] == 0)
			{
				line.side = 1;
			}
			if (board.grid[x][y] != board.grid[x - 1][y])
			{
				for (int i = 1; i < left; i++)
				{
					if (board.grid[x][y] == board.grid[x + i][y])
					{
						line.pos = 1;
						line.length += 1;
					}
					else if (board.grid[x + i][y] == 0)
					{
						line.side += 1;
						break;
					}
					else
					{
						break;
					}

				}
			}
			else
			{
				line.pos = 1;
			}
			//only push single chess at first time, and reset length and side
			if (line.length > 1)
			{
				lines.push_back(line);
			}
			line.length = 1;
			line.side = 0;
			if (board.grid[x][y - 1] == 0)
			{
				line.side = 1;
			}
			if (board.grid[x][y] != board.grid[x][y - 1])
			{
				for (int i = 1; i < down; i++)
				{
					if (board.grid[x][y] == board.grid[x][y + i])
					{
						line.pos = 2;
						line.length += 1;
					}
					else if (board.grid[x][y + i] == 0)
					{
						line.side += 1;
						break;
					}
					else
					{
						break;
					}
				}
			}
			else
			{
				line.pos = 2;
			}
			if (line.length > 1)
			{
				lines.push_back(line);
			}

			line.length = 1;
			line.side = 0;
			if (board.grid[x - 1][y - 1] == 0)
			{
				line.side = 1;
			}
			if (board.grid[x][y] != board.grid[x - 1][y - 1])
			{
				for (int i = 1; i < slant1; i++)
				{
					if (board.grid[x][y] == board.grid[x + i][y + i])
					{
						line.pos = 3;
						line.length += 1;
					}
					else if (board.grid[x + i][y + i] == 0)
					{
						line.side += 1;
						break;
					}
					else
					{
						break;
					}
				}
			}
			else
			{
				line.pos = 3;
			}
			if (line.length > 1)
			{
				lines.push_back(line);
			}

			line.length = 1;
			line.side = 0;
			if (board.grid[x - 1][y + 1] == 0)
			{
				line.side = 1;
			}
			if (board.grid[x][y] != board.grid[x - 1][y + 1])
			{
				for (int i = 1; i < slant2; i++)
				{
					if (board.grid[x][y] == board.grid[x + i][y - i])
					{
						line.pos = 4;
						line.length += 1;
					}
					else if (board.grid[x + i][y - i] == 0)
					{
						line.side += 1;
						break;
					}
					else
					{
						break;
					}
				}
			}
			else
			{
				line.pos = 4;
			}
			if (line.length > 1)
			{
				lines.push_back(line);
			}

			if (line.pos == 0)
			{
				line.side = 0;
				if (board.grid[x - 1][y] == 0)
				{
					line.side += 1;
				}
				if (board.grid[x - 1][y - 1] == 0)
				{
					line.side += 1;
				}
				if (board.grid[x - 1][y + 1] == 0)
				{
					line.side += 1;
				}
				if (board.grid[x][y - 1] == 0)
				{
					line.side += 1;
				}
				if (board.grid[x][y + 1] == 0)
				{
					line.side += 1;
				}
				if (board.grid[x + 1][y] == 0)
				{
					line.side += 1;
				}
				if (board.grid[x + 1][y - 1] == 0)
				{
					line.side += 1;
				}
				if (board.grid[x + 1][y + 1] == 0)
				{
					line.side += 1;
				}
				lines.push_back(line);
			}
		}
	}
}