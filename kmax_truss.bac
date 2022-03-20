#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <chrono>
#include <sys/time.h>

using namespace std;

typedef chrono::high_resolution_clock Clock;

const int NUM_THREADS_THIS_COMPUTER = omp_get_num_procs();


class TimeInterval{
public:
    TimeInterval(){
        check();
    }
    void check(){
        gettimeofday(&tp, NULL);
    }
    void print(const char* title){
        struct timeval tp_end, tp_res;
        gettimeofday(&tp_end, NULL);
        timersub(&tp_end, &tp, &tp_res);
        cout << title << ": ";
        printf("%ld.%06lds\n", tp_res.tv_sec, tp_res.tv_usec);
    }
private:
    struct timeval tp;
};

#define  DEBUG_DF

TimeInterval allTime;
TimeInterval calcTime;
TimeInterval nowTime[10];
TimeInterval tmpTime;



int64_t *calc_rowstart(int *edgelist_two, int64_t nodeNum, int64_t numEdges);
int64_t *calc_nodeDegree(int64_t *row_start, int64_t nodeNum);
int64_t calc_maxSuitKcore(int64_t *nodeDegree, int64_t nodeNum);
int64_t calc_kmax_core(int64_t nodeNum, int64_t numEdges, int *edgelist_two, int64_t * row_start, int64_t *nodeDegree, int64_t *&nodeDelete);
void parallelmemcpy(int *&news, int *ori, int64_t n);
void parallelmemcpy_64(int64_t *&news, int64_t *ori, int n);
int64_t calc_ktry_core_graph(int64_t nodeNum_origin,int64_t numEdges_origin, int *edgelist_two_origin, int64_t *&nodeDelete, int64_t k_try);
void reconstruct_Graph(int64_t *nodeNum_re, int64_t *numEdges_re, int *edgelist_two_re, int64_t *nodeDelete_re);
int64_t calc_kmax_core(int64_t k_floor, int64_t nodeNum, int64_t numEdges, int *edgelist_two, int64_t *&nodeDelete);
int64_t binary_search_(int64_t l,int64_t r,int64_t w, int *edgelist_two);
int64_t count_tris(int64_t *edge_trianglenum, int *endpoints, int64_t n_edge, int64_t n_node, int64_t n_try);
int64_t ktruss_chk(int *edgelist_two, int64_t nodeNum, int64_t numEdges, int64_t k_try);



// 计算csr格式行首起始位置
int64_t * calc_rowstart(int *edgelist_two, int64_t nodeNum, int64_t numEdges)
{
    //共有nodeNum大小
    int64_t *row_start_tmp = new int64_t[nodeNum+1]();
    row_start_tmp[0]=0;
    #pragma omp parallel for
    for (int64_t i = 1; i < numEdges; i++)
    {
        //判断当前出发点是否大于前一个，索引2和0对比
        //因为不是计数，所以不用原子操作，只有前后不一样的时候才会赋值，不会冲突
        //i 对应（edgelist_two[i*2] ，edgelist_two[i*2+1]）
        //1 99 1 100 2 45 2 77 2 99 3 87 5 98 5 99
        //i=1,if([2]>[0])(1不大于1)
        //i=2,if([4]>[2])(2大于1) for(1+1 <= 2) row_start[2] = i;
        //...
        //i=6,if([12]>[10])(5大于3) for(3+1 <= 5) row_start[4,5] = i;
        //0 1 2 3 4 5 索引,共5+1长
        //0 0 2 5 6 6 出发点起始坐标
        if (edgelist_two[i*2] > edgelist_two[(i-1)*2])
        {
            for (int64_t k = (edgelist_two[(i-1)*2]+1); k <= edgelist_two[i*2]; k++) 
                row_start_tmp[k] = i;
        }
    }

    //cout<< "nodeNum - (edgelist_two[(numEdges-1)*2]+1) "<< nodeNum << "-" <<(edgelist_two[(numEdges-1)*2]+1)<<endl;

    for (int64_t i = (edgelist_two[(numEdges-1)*2]+1); i <= nodeNum; i++)
        row_start_tmp[i] = numEdges;
    row_start_tmp[nodeNum] = numEdges;

    return row_start_tmp;
}

//计算节点度数
int64_t * calc_nodeDegree(int64_t *row_start, int64_t nodeNum){
    int64_t *nodeDegree_tmp = new int64_t[nodeNum]();

    //每行的终止坐标减去起始坐标就是度数
    #pragma omp parallel for
    for (int64_t i = 0; i < nodeNum; i++)
    {
        nodeDegree_tmp[i] = row_start[i+1] - row_start[i];
    }
    return nodeDegree_tmp;
}



void parallelmemcpy(int *&news, int *ori, int64_t n){
    #pragma omp parallel for
    for (int64_t i_n = 0; i_n < n; i_n++){ news[i_n] = ori[i_n]; }
}

void parallelmemcpy_64(int64_t *&news, int64_t *ori, int64_t n){
    #pragma omp parallel for
    for (int64_t i_n = 0; i_n < n; i_n++){ news[i_n] = ori[i_n]; }
}



/*  尝试并减少kcore图的规模 */
//传进去值，返回delete_node数组和删掉了多少边
int64_t calc_ktry_core_graph(int64_t nodeNum_origin,int64_t numEdges_origin, int *edgelist_two_origin, int64_t *&nodeDelete, int64_t k_try){

    /* 计算新的行下标数组 */
    /*计算第i个出发点在第几个位置，假设在x位，那么位置为（x*2，x*2+1）*/
    int64_t *row_start_origin = calc_rowstart(edgelist_two_origin, nodeNum_origin, numEdges_origin);

    /* 数组大小为节点度数，记录每个结点的度 */
    int64_t *deg_origin = calc_nodeDegree(row_start_origin, nodeNum_origin);

    int64_t k = k_try;  //当前k值
    int64_t node_delete = 0;  // 删除的节点数

#ifdef DEBUG_DF
    cout << "   calc_ktry_core_graph : K_try: " << k_try<< "  nodeNum: "<< nodeNum_origin << endl;
#endif
    
  
    while(true)  //循环删除不符合条件得边
    {
        int64_t flag = 0;  //如果遍历所有边都符合条件，则==0 跳出循环
        // printf("nodeNum_origin=%d \n",nodeNum_origin);
        #pragma omp parallel for reduction(+:node_delete) schedule(dynamic, 128) 
        for(int64_t i = 0; i < nodeNum_origin; i++)
        {   
            if(nodeDelete[i] == 0 && deg_origin[i] < k) //如果这个点还没被删除，并且度数小于k
            {
                nodeDelete[i] = 1;  //标记该点要被删除
                flag = 1;           //此轮不跳出循环
                node_delete++;      //要删除的边个数加加
                int64_t rn = row_start_origin[i+1];  //遍历i的所有邻居
                for(int64_t j = row_start_origin[i]; j < rn; j++)
                    #pragma omp atomic
                    deg_origin[edgelist_two_origin[j*2+1]] --; //原子减减，邻居的度
            }   
        }
        if(flag == 0)  //如果遍历所有边都符合条件，则==0 跳出循环
            break;
    }
  
#ifdef DEBUG_DF
    cout << "   calc_ktry_core_graph : node_delete: " << node_delete << "  nodeNum: "<< nodeNum_origin << endl;
#endif
    
    return node_delete;
}

void reconstruct_Graph(int64_t *nodeNum_re, int64_t *numEdges_re, int *edgelist_two_re, int64_t *nodeDelete_re){
TimeInterval cmpTime;
#ifdef DEBUG_DF
    printf("   开始边数筛选\n");
#endif
    
    /* 数据替换与重排编号 */
    // auto datachange1 = Clock::now(); 

    int64_t numEdges_tmp = 0;
    int64_t ori_numedges = *numEdges_re;

    int *tempedgelist = new int[ori_numedges*2];
    parallelmemcpy(tempedgelist, edgelist_two_re, ori_numedges*2);

    //任务均衡
    int64_t * task_balance = new int64_t[NUM_THREADS_THIS_COMPUTER+1]();
    int64_t * task_edge = new int64_t[NUM_THREADS_THIS_COMPUTER+1]();

    for (int64_t i = 0; i <= NUM_THREADS_THIS_COMPUTER; ++i)
    {
        task_balance[i]=0;
        task_edge[i]=0;
    }
    
    int64_t steplen = (ori_numedges)/NUM_THREADS_THIS_COMPUTER;
    //50/24=2     0 2 4 6  ``````  49
    for (int64_t i = 1; i < NUM_THREADS_THIS_COMPUTER; ++i)
    {
        task_balance[i]=task_balance[i-1]+(steplen);
    }
    task_balance[NUM_THREADS_THIS_COMPUTER]=ori_numedges;

    #pragma omp parallel for
    for (int64_t i = 0; i < NUM_THREADS_THIS_COMPUTER; ++i)
    {
        for (int64_t j = task_balance[i]; j < task_balance[i+1]; ++j)
        {
            int l=edgelist_two_re[2*j];
            int r=edgelist_two_re[2*j+1];
            if (nodeDelete_re[l] != 1 && nodeDelete_re[r] != 1){
                task_edge[i+1]++;
            }
        }
    }

    for (int64_t i = 0; i <= NUM_THREADS_THIS_COMPUTER; ++i) numEdges_tmp+=task_edge[i];
    for (int64_t i = 1; i <= NUM_THREADS_THIS_COMPUTER; ++i)
    {
        task_edge[i]+=task_edge[i-1];
    }

    // 把没有被删除的边给往前挪一挪
    #pragma omp parallel for
    for (int64_t i = 0; i < NUM_THREADS_THIS_COMPUTER; ++i)
    {
        int64_t start=task_edge[i];

        for (int64_t j = task_balance[i]; j < task_balance[i+1]; ++j)
        {
            int l=tempedgelist[2*j];
            int r=tempedgelist[2*j+1];
            if (nodeDelete_re[l] != 1 && nodeDelete_re[r] != 1){
                edgelist_two_re[2*start] = l;
                edgelist_two_re[2*start+1] = r;
                start++;
            }
        }
        // printf("i = %d %d %d \n", i, start, task_edge[i+1]);
    }
    
    int64_t orinode = *nodeNum_re;
    int64_t *rename_list = new int64_t [orinode]();
    int64_t n_node_tmp = 0;
    for (int i = 0; i < orinode; i++)
    {
        if (!nodeDelete_re[i])
        {
            rename_list[i] = n_node_tmp;
            n_node_tmp++;
        }
    }

    #pragma omp parallel for
    for (int64_t i = 0; i < numEdges_tmp*2; i++)
    {
        edgelist_two_re[i] = rename_list[edgelist_two_re[i]];
    }
    delete [] rename_list;


    delete [] task_balance;
    delete [] task_edge;
    delete [] tempedgelist;


    //释放部分数组大小，两个值交换的思想
    // int *edgelist_two_new = new int[numEdges_tmp*2];
    // parallelmemcpy(edgelist_two_new, edgelist_two_re, numEdges_tmp*2);
    // int *edgelist_two_tmp = edgelist_two_re;
    // edgelist_two_re = edgelist_two_new;
    // delete [] edgelist_two_tmp;

    *numEdges_re = numEdges_tmp;
    *nodeNum_re = n_node_tmp;

#ifdef DEBUG_DF
    cmpTime.print("   data reduce time = ");
    cmpTime.check();
    printf( "   subgraph Edge = %d and Node = %d \n", numEdges_tmp, n_node_tmp);  
    printf("   ******************************************************\n");
#endif

}



int64_t calc_kmax_core(int64_t k_floor, int64_t nodeNum, int64_t numEdges, int *edgelist_two, int64_t *&nodeDelete)
{
    /* 计算新的行下标数组 */
    /*计算第i个出发点在第几个位置，假设在x位，那么位置为（x*2，x*2+1）*/
    int64_t *row_start = calc_rowstart(edgelist_two, nodeNum, numEdges);

    /* 数组大小为节点度数，记录每个结点的度 */
    int64_t *nodeDegree = calc_nodeDegree(row_start, nodeNum);

    int64_t k = k_floor;  //当前k值
    //int k_step = pow(nodeNum,1./2); //最大k值  k每次增加的幅度
    int64_t k_step = 32; //最大k值  k每次增加的幅度
    int64_t node_delete = 0, node_delete_backup = 0;  // 删除的节点数 删除的节点数备份 
    int64_t *nodeDelete_backup = new int64_t[nodeNum];  //nodeDelete（要删除的边）的备份
    int64_t *deg_backup = new int64_t[nodeNum];   // 度数的备份

    while(node_delete < nodeNum)  //当删除的节点小于总结点
    {
        // backup
        if(k_step > 1)  //当你大步走的时候
        {
            node_delete_backup = node_delete; //备份删除的边数目
            parallelmemcpy_64(deg_backup, nodeDegree, nodeNum);
        }

        parallelmemcpy_64(nodeDelete_backup, nodeDelete, nodeNum);

#ifdef DEBUG_DF
    cout << "   k: " << k << "  node_delete: " << node_delete << "  nodeNum: "<< nodeNum << "  nodeleft: "<< nodeNum-node_delete<< endl;
#endif

        k += k_step;
        while(true)  //循环删除不符合条件得点
        {
            int64_t flag = 0;  //如果遍历所有点都符合条件，则==0 跳出循环
            #pragma omp parallel for reduction(+:node_delete) schedule(dynamic) 
            for(int64_t i = 0;i < nodeNum;i++)
            {   
                if(nodeDelete[i] == 0 && nodeDegree[i] < k) //如果这个点还没被删除，并且度数小于k
                {
                    nodeDelete[i] = 1;  //标记该点要被删除
                    flag = 1;           //此轮不跳出循环
                    node_delete++;      //要删除的边个数加加
                    int64_t rn = row_start[i+1];  //遍历i的所有邻居
                    for(int64_t j = row_start[i]; j < rn; j++)
                        #pragma omp atomic
                        nodeDegree[edgelist_two[j*2+1]] --; //原子减减，邻居的度
                }
            }
            if(flag == 0)  //如果遍历所有边都符合条件，则==0 跳出循环
                break;
        }
        if(nodeNum - node_delete < k + 1  and k_step > 1)  //如果删除的点=总点数，本轮删点结束，k_step
        {
            k-=k_step;k_step = 1;
            node_delete = node_delete_backup;
            parallelmemcpy_64(nodeDegree, deg_backup, nodeNum);
            parallelmemcpy_64(nodeDelete, nodeDelete_backup, nodeNum);
        }
    }

#ifdef DEBUG_DF
    cout << "   finish over kmax k: " << k << "  node_delete: " << node_delete << "  nodeNum: "<< nodeNum << " nodeleft: "<<nodeNum-node_delete<< endl;
#endif

    parallelmemcpy_64(nodeDelete, nodeDelete_backup, nodeNum);
    delete [] deg_backup;  //删除度数的备份
    delete [] nodeDelete_backup;
    return k-1;
}


int64_t binary_search_(int64_t l,int64_t r,int64_t w, int *edgelist_two){
    int64_t left =l,right=r-1;
    while(left<=right)
    {
        int64_t mid =(left+right)/2;
        if(w==edgelist_two[mid*2+1]) return mid;
        if(w>edgelist_two[mid*2+1]) left=mid+1;
        else right =mid-1;
    } return -1;
}

// 计算每条边所属三角形的数量
//                            
int64_t count_tris(int64_t *edge_trianglenum, int *endpoints, int64_t n_edge, int64_t n_node, int64_t n_try)
{

    /*计算第i个出发点在第几个位置，假设在x位，那么位置为（x*2，x*2+1）*/
    int64_t *row_start = calc_rowstart(endpoints, n_node, n_edge);

    int64_t tc_numThread = omp_get_max_threads();
    char *vld = new char [tc_numThread*n_node](); //vld 大小为节点数乘以线程数，像是私有化
    
    bool drop_flag = false;
    // 有所有点对应的集合求交, 对填充率低的利比较策略，对填充高的用利用原子加
    #pragma omp parallel for schedule(dynamic) 
    for (int64_t i = 0; i < n_node; i++) //遍历所有顶点
    {
        //取第i个顶点的邻居
        int64_t i_left_index = row_start[i];
        int64_t i_right_index = row_start[i+1];
        int64_t i_nbr=i_right_index-i_left_index;

        int64_t numThread_id = omp_get_thread_num();  // 获取当前线程id
        char *vl_ini = vld + numThread_id*n_node; // 

        int64_t k, z;
        for (k = i_left_index; k < i_right_index; k++)
            *(vl_ini + endpoints[2*k+1]) = 1;

        int64_t triangle_count;
        for (k = i_left_index; k < i_right_index; k++)
        {
            int64_t i_neibor_name = endpoints[2*k+1]; 
            int64_t i_nbr_nbr =row_start[i_neibor_name+1]-row_start[i_neibor_name];
            // u < v
            //if(i < i_neibor_name) continue;
            if(i_nbr < i_nbr_nbr){
                continue;
            }
            else if(i_nbr == i_nbr_nbr && i < i_neibor_name){
                continue;
            }

            triangle_count = 0;
            // 遍历i的所有邻居的邻居
            for (z = row_start[i_neibor_name]; z < row_start[i_neibor_name+1]; z++)
                triangle_count += *(vl_ini + endpoints[2*z+1]);

            edge_trianglenum[k] = triangle_count;
            int64_t x = binary_search_(row_start[i_neibor_name],row_start[i_neibor_name+1],i,endpoints);
            if(x!=-1) edge_trianglenum[x] = triangle_count;

            if (triangle_count < n_try -2 ) {
                drop_flag = true;
            }
        }

        for (k = i_left_index; k < i_right_index; k++)
            *(vl_ini + endpoints[2*k+1]) = 0;

    }

    delete [] vld;
    delete [] row_start;

    if (drop_flag)
    {

        int64_t numEdges_tmp = 0;
        int64_t ori_numedges = n_edge;
        int *tempedgelist = new int[ori_numedges*2];
        parallelmemcpy(tempedgelist, endpoints, ori_numedges*2);

        //任务均衡
        int64_t * task_balance = new int64_t[NUM_THREADS_THIS_COMPUTER+1]();
        int64_t * task_edge = new int64_t[NUM_THREADS_THIS_COMPUTER+1]();
        #pragma omp parallel for
        for (int64_t i = 0; i <= NUM_THREADS_THIS_COMPUTER; ++i)
        {
            task_balance[i]=0;
            task_edge[i]=0;
        }
        
        int64_t steplen = (ori_numedges)/NUM_THREADS_THIS_COMPUTER;
        //50/24=2     0 2 4 6  ``````  49
        for (int64_t i = 1; i < NUM_THREADS_THIS_COMPUTER; ++i)
        {
            task_balance[i]=task_balance[i-1]+(steplen);
        }
        task_balance[NUM_THREADS_THIS_COMPUTER]=ori_numedges;

        #pragma omp parallel for
        for (int64_t i = 0; i < NUM_THREADS_THIS_COMPUTER; ++i)
        {
            for (int64_t j = task_balance[i]; j < task_balance[i+1]; ++j)
            {
                if (edge_trianglenum[j] >= n_try-2 ){
                    task_edge[i+1]++;
                }
            }
        }

        for (int64_t i = 0; i <= NUM_THREADS_THIS_COMPUTER; ++i) numEdges_tmp+=task_edge[i];
        for (int64_t i = 1; i <= NUM_THREADS_THIS_COMPUTER; ++i)
        {
            task_edge[i]+=task_edge[i-1];
        }

        // 把没有被删除的边给往前挪一挪
        #pragma omp parallel for
        for (int64_t i = 0; i < NUM_THREADS_THIS_COMPUTER; ++i)
        {
            int64_t start=task_edge[i];

            for (int64_t j = task_balance[i]; j < task_balance[i+1]; ++j)
            {
                if (edge_trianglenum[j] >= n_try-2 ){
                    endpoints[2*start] = tempedgelist[2*j];
                    endpoints[2*start+1] = tempedgelist[2*j+1];
                    start++;
                }
            }
            // printf("i = %d %d %d \n", i, start, task_edge[i+1]);
        }

        return numEdges_tmp;
    }
    else
        return n_edge;

}


// 检查压缩全过程
int64_t ktruss_chk(int *edgelist_two, int64_t nodeNum, int64_t numEdges, int64_t k_try)
{
#ifdef DEBUG_DF
    printf("   ");
#endif
    

    int *edges_cp = new int [numEdges*2];
    parallelmemcpy(edges_cp, edgelist_two, numEdges*2);
    int64_t *edge_trianglenum = new int64_t [numEdges];

    int64_t numEdges_cp;
    int64_t numEdges_rem = numEdges;
    do
    {
        numEdges_cp = numEdges_rem;
        numEdges_rem = count_tris(edge_trianglenum, edges_cp, numEdges_cp, nodeNum, k_try);

#ifdef DEBUG_DF
    printf("\u25A0");
#endif

    } while ((numEdges_cp > numEdges_rem) && (numEdges_rem > k_try-2));

#ifdef DEBUG_DF
    printf("\n");
#endif

    if (numEdges_rem > k_try-2){

        memcpy(edgelist_two, edges_cp, sizeof(int)*numEdges_rem*2);
    }

    delete [] edges_cp;
    delete [] edge_trianglenum;

    return numEdges_rem;
}



int main(int argc,char *argv[]) {
    
    omp_set_num_threads(NUM_THREADS_THIS_COMPUTER);
    // auto startTime = Clock::now(); 
    
    /* 判断读入的参数格式是否正确 */
    char mtxName[100],dataKind[100];
    memset(mtxName, '\0', sizeof(mtxName));
    if (argc == 3)
    {
        strcpy(mtxName,argv[2]);
        printf("Martrix file name: %s \n",mtxName);
        strcpy(dataKind,argv[1]);
        
    }
    else{
        cout << "用法: ./k-truss-cpu -3f XXX.csv  or  ./k-truss-cpu -2f XXX.csv" << endl;
        exit(0);
    }



    /* 利用vmtouch把文件读取到内存中 */
    //system("vmtouch -vt ../ktruss-data/soc-LiveJournal.tsv");

    /* 通过内存映射实现快速读取 */
    char *rawdata = NULL;
    int fd=open(mtxName,O_RDONLY);
    int64_t size = lseek(fd, 0, SEEK_END);


#ifdef DEBUG_DF
    printf("Martrix file size: %lld \n",size);
#endif
    rawdata = (char *) mmap( NULL, size ,PROT_READ, MAP_SHARED, fd, 0 );

    /* 根据\n统计总边数 */
    int64_t numEdges=0;
    #pragma omp parallel for reduction(+:numEdges)
    for (int64_t i = 0; i < size; i++)
    {
        if (rawdata[i] == '\n') 
            numEdges++;
    }

    //存储边的数组,下标01表示第一条边的出发结束
    int *edgelist_two = new int [numEdges*2]();

    /* 按照线程大概分块，确定每一块数据以\n结束，存储第i块尾部\n的下标 */
    // 假设\n下标为4 8 12 16 分成4部分
    // 要存储 0 5 9 13 16 存5个
    int64_t *edge_part = new int64_t [NUM_THREADS_THIS_COMPUTER+1]();//假设开4
    edge_part[0]=0;
    #pragma omp parallel for
    for (int64_t i = 1; i < NUM_THREADS_THIS_COMPUTER; i++)//遍历1-3，edge_part赋值0-2
    {
        int64_t edge_part_end = (size/NUM_THREADS_THIS_COMPUTER)*i;//粗略指定分割位置
        while (rawdata[edge_part_end] != '\n')//微调到每一部分结束都是\n
            edge_part_end++;
        //edge_part_end此时等于\n的位置
        edge_part_end++;
        //edge_part_end此时等于\n+1的位置
        edge_part[i] = edge_part_end;
    }
    //edge_part给下标3赋值size，即rawdata数组超出一位，后边是<号
    edge_part[NUM_THREADS_THIS_COMPUTER] = size;


    /* 计算每一段的个数，按照每一段的\n计算 */
    // 0 5 9 13 16
    // 第一段0-5 NUM_THREADS_THIS_COMPUTER+1]();//存放每部分有多少条边
    int64_t *edge_part_num = new int64_t [NUM_THREADS_THIS_COMPUTER+1]();//存放每一组都有多少条边
    #pragma omp parallel for
    for (int64_t i = 1; i < NUM_THREADS_THIS_COMPUTER; i++)//遍历1-3，edge_part_num赋值0-2
    {
        int64_t npz = 0;
        for (int64_t z = edge_part[i-1]; z < edge_part[i]; z++)//edge_part[0]默认是0，被当作起始位置
        {
            if (rawdata[z] == '\n')
                npz++;
        }
        edge_part_num[i] = npz*2;
    }
    
    /* 计算每部分在边数组中起始索引，方便并行 */
    for (int64_t i = 1; i <= NUM_THREADS_THIS_COMPUTER; i++)// 0 1 2 3 0  --> 0 1 3 6 
        edge_part_num[i] += edge_part_num[i-1];//累加是为了计算每一段的起始位置

    /* 多线程并行将文本数据转换成数组 */
    #pragma omp parallel for
    for (int64_t i = 0; i < NUM_THREADS_THIS_COMPUTER; i++)//从0开始，没有处理最后一个
    {
        int64_t part_start = edge_part_num[i];//每一块的起始位置，因此不用计算最后一块的大小，最后一块的起始位置为其他几块之和
        for (int64_t z = edge_part[i]; z < edge_part[i+1];)//范围，一块的大小
        {
            int tmpValue1=0,tmpValue2=0;
            char ch=rawdata[z++];
            
            while(ch>='0'&&ch<='9') tmpValue1=tmpValue1*10+ch-'0',ch=rawdata[z++];
            while(ch<'0'||ch>'9'){ch=rawdata[z++];}
            while(ch>='0'&&ch<='9') tmpValue2=tmpValue2*10+ch-'0',ch=rawdata[z++];
            edgelist_two[part_start++]=tmpValue2;
            edgelist_two[part_start++]=tmpValue1;
            while(ch<'0'||ch>'9'){ch=rawdata[z++];}
            while(ch>='0'&&ch<='9'){ch=rawdata[z++];}
        }
    }

    //计算节点数
    int64_t nodeNum = edgelist_two[(numEdges-1)*2];

    if(edgelist_two[0] == 1){
        #pragma omp parallel for
        for (int64_t i = 0; i < numEdges; i++){
            edgelist_two[i*2]--;
            edgelist_two[i*2+1]--;
            // if(edgelist_two[i*2] < 0){
            //     printf("edgelist_two[%d] = %d \n",i*2,edgelist_two[i*2]);
            // }
            // if(edgelist_two[i*2+1] < 0){
            //     printf("edgelist_two[%d] = %d \n",i*2+1,edgelist_two[i*2+1]);
            // }
        }
    }


    delete [] edge_part;
    delete [] edge_part_num;
    // 关闭镜像通道
    munmap(rawdata, size);
    close(fd);

#ifdef DEBUG_DF
    tmpTime.print("一、 Read file time equals");
    tmpTime.check();
    nowTime[0].print("程序已进行");
    nowTime[0].check();
    printf( "   Edge num equals  %d and Node num equals %d \n",numEdges/2,nodeNum); 
#endif

    calcTime.check();

    // /* 计算k-core上界 */
     int maxSuitKcore = nodeNum-1;//calc_maxSuitKcore(nodeDegree, nodeNum);

#ifdef DEBUG_DF
    printf( "   max Suit Kcore equals  %d \n",maxSuitKcore);
#endif

    /* 备份原图和边点数据 */
    int64_t numEdges_sub = numEdges;
    int *edges_sub = new int [numEdges_sub*2];
    int64_t nodeNum_sub = nodeNum;
    parallelmemcpy(edges_sub, edgelist_two, numEdges*2);
    int64_t *nodeDelete = new int64_t[nodeNum]();


    /* 不一定会备份kmax-1子图，这里备份2个小图，k=50 和 100 */
    int64_t numEdges_sub_30 = 0;
    int *edges_sub_30;
    int64_t nodeNum_sub_30 = 0;

    int64_t numEdges_sub_100 = 0;
    int *edges_sub_100;
    int64_t nodeNum_sub_100 = 0;



    //在子图进行k-core的子图缩减
    //尝试不同k值，稍微缩减kcore子图规模
    int64_t k_try = 0;
    for (int64_t k = 30; k < nodeNum-1; k+=70)
    {
        k_try = k;
        int64_t nodeDelete_num = calc_ktry_core_graph(nodeNum_sub, numEdges_sub, edges_sub, nodeDelete, k_try);
        //还剩余边的话
        if(nodeDelete_num < nodeNum_sub){

#ifdef DEBUG_DF
    printf("   k_try = %d ,now k_try < max. \n", k_try);
    printf("   ******************************************************\n");
#endif
            if(nodeDelete_num >= nodeNum_sub*0.2)
                reconstruct_Graph(&nodeNum_sub, &numEdges_sub, edges_sub, nodeDelete);

            if(k_try == 30) {
                numEdges_sub_30 = numEdges_sub;
                nodeNum_sub_30 = nodeNum_sub;
                edges_sub_30 = new int [numEdges_sub_30*2];
                parallelmemcpy(edges_sub_30, edges_sub, numEdges_sub_30*2);
            }
            else if (k_try == 100)
            {
                numEdges_sub_100 = numEdges_sub;
                nodeNum_sub_100 = nodeNum_sub;
                edges_sub_100 = new int [numEdges_sub_100*2];
                parallelmemcpy(edges_sub_100, edges_sub, numEdges_sub_100*2);
            }


            #pragma omp parallel for
            for (int64_t i = 0; i < nodeNum; i++) nodeDelete[i] = 0;
        }
        else{ //边都被删完了

#ifdef DEBUG_DF
    printf("   k_try = %d ,now k_try >= max. \n", k_try);
#endif
            
            k_try-=70;
            if(k_try < 3 ) k_try = 3;
            break;
        }
    }

#ifdef DEBUG_DF
    printf("   备份half-kcore =%d 小图\n",k_try/2);
#endif

//再备份一份k-core/2小图
    int64_t k_half = k_try/2;
    int64_t numEdges_half = 0;
    int64_t nodeNum_half = 0;
    int *edgelist_two_half;

    #pragma omp parallel for
    for (int64_t i = 0; i < nodeNum; i++) nodeDelete[i] = 0;

    if (k_half > 100 && numEdges_sub_100 != 0)
    {

#ifdef DEBUG_DF
    printf("   存在偏小图备份，直接使用备份k=100\n");
#endif
        numEdges_half = numEdges_sub_100;
        nodeNum_half = nodeNum_sub_100;
        edgelist_two_half = new int [numEdges_half*2];
        parallelmemcpy(edgelist_two_half, edges_sub_100, numEdges_half*2);

        numEdges_sub_30 = 0;
        nodeNum_sub_30 = 0;
        delete [] edges_sub_30;

    }
    else if(k_half > 30 && numEdges_sub_30 != 0) {

#ifdef DEBUG_DF
    printf("   存在偏小图备份，直接使用备份k=30\n");
#endif

        numEdges_half = numEdges_sub_30;
        nodeNum_half = nodeNum_sub_30;
        edgelist_two_half = new int [numEdges_half*2];
        parallelmemcpy(edgelist_two_half, edges_sub_30, numEdges_half*2);


    }

    if ( numEdges_half != 0 )
    {
        calc_ktry_core_graph(nodeNum_half, numEdges_half, edgelist_two_half, nodeDelete, k_half);
        reconstruct_Graph(&nodeNum_half, &numEdges_half, edgelist_two_half, nodeDelete);
    }


#ifdef DEBUG_DF
    printf("   max k_try = %d ,now k_try < maxSuitKcore. subgraph edge = %d nodeNum = %d. \n",k_try,numEdges_sub,nodeNum_sub);
    tmpTime.print("二、 k-core 子图缩减结束，时间 = ");
    tmpTime.check();
    nowTime[1].print("程序已进行");
    nowTime[1].check();
#endif


    #pragma omp parallel for
    for (int64_t i = 0; i < nodeNum_sub; i++) nodeDelete[i] = 0;

    /* 计算kmax-core */

    int64_t k_core_max = calc_kmax_core(k_try, nodeNum_sub, numEdges_sub, edges_sub, nodeDelete);
    reconstruct_Graph(&nodeNum_sub, &numEdges_sub, edges_sub, nodeDelete);


#ifdef DEBUG_DF
    tmpTime.print("三、 计算k-core 结束，时间 = ");
    tmpTime.check();
    nowTime[2].print("程序已进行");
    nowTime[2].check();
    printf( "k_core_max equals %d nodeNum_sub = %d  numEdges_sub = %d \n",k_core_max,nodeNum_sub,numEdges_sub);  
#endif



    // edge_left 为修剪后还剩下的边数
    int64_t edge_left, try_floor = 3, try_ceil = k_core_max+1;

#ifdef DEBUG_DF
    printf("   kcore 上的ktruss区间为 [%d, %d]. \n", try_floor, try_ceil);  
#endif
    
    do
    {
        k_try = (try_floor+try_ceil)/2;
        edge_left = ktruss_chk(edges_sub, nodeNum_sub, numEdges_sub, k_try);

#ifdef DEBUG_DF
    cout << "   检查数 " << k_try << " 完成，剩余边数为 " << edge_left << endl;
#endif

        if (edge_left > k_try)
        {
            numEdges_sub = edge_left;
            try_floor = k_try;
        }
        else{
            try_ceil = k_try;
        }

    }
    while (try_ceil - try_floor > 1);

    
 #ifdef DEBUG_DF
    tmpTime.print("四、 计算k-core上的ktruss下界 结束，时间 = ");
    tmpTime.check();
    nowTime[3].print("程序已进行");
    nowTime[3].check();
    printf("kmaxcore上的kmaxtruss为 %d, 剩余的边数为 %d. \n", try_floor, numEdges_sub/2); 
#endif

    int64_t numEdges_final = numEdges;
    int64_t nodeNum_final = nodeNum;
    int *edgelist_two_final = edgelist_two;

    #pragma omp parallel for
    for (int64_t i = 0; i < nodeNum; i++) nodeDelete[i] = 0;

    if (try_floor-1 > k_half && numEdges_half != 0)
    {
#ifdef DEBUG_DF
    printf("   存在偏小图备份，直接使用备份half-core = %d \n",k_half);
#endif
        numEdges_final = numEdges_half;
        nodeNum_final = nodeNum_half;
        edgelist_two_final = edgelist_two_half;
    }
    else if (try_floor-1 > 100 && numEdges_sub_100 != 0)
    {
#ifdef DEBUG_DF
    printf("   存在偏小图备份，直接使用备份k=100\n");
#endif
        numEdges_final = numEdges_sub_100;
        nodeNum_final = nodeNum_sub_100;
        edgelist_two_final = edges_sub_100;
    }
    else if(try_floor-1 > 30 && numEdges_sub_30 != 0) {
#ifdef DEBUG_DF
    printf("   存在偏小图备份，直接使用备份k=30\n");
#endif
        numEdges_final = numEdges_sub_30;
        nodeNum_final = nodeNum_sub_30;
        edgelist_two_final = edges_sub_30;
    }


    calc_ktry_core_graph(nodeNum_final, numEdges_final, edgelist_two_final, nodeDelete, try_floor-1);
    reconstruct_Graph(&nodeNum_final, &numEdges_final, edgelist_two_final, nodeDelete);


#ifdef DEBUG_DF
    tmpTime.print("五、 从大图获取kmax-1-core子图 结束，时间 = ");
    tmpTime.check();
    nowTime[4].print("程序已进行");
    nowTime[4].check();
    printf( "   k_core-1 equals %d nodeNum_final = %d  numEdges_final = %d \n",try_floor-1,nodeNum_final,numEdges_final);  
#endif


    // edge_left 为修剪后还剩下的边数
    edge_left = numEdges_final;
    try_ceil = k_core_max+1;
    k_try = try_floor;

#ifdef DEBUG_DF
    printf("   k-1-core上的ktruss区间为 [%d, %d]. \n", try_floor, try_ceil);
#endif

    
    do
    {
        edge_left = ktruss_chk(edgelist_two_final, nodeNum_final, numEdges_final, k_try);
        
#ifdef DEBUG_DF
    cout << "   检查数 " << k_try << " 完成，剩余边数为 " << edge_left << endl;
#endif
        if (edge_left > k_try)
        {
            numEdges_final = edge_left;
            try_floor = k_try;
        }
        k_try++;

    }
    while (edge_left > k_try);


#ifdef DEBUG_DF
    tmpTime.print("六、 计算k-1-core上的ktruss结束，时间 =");
    tmpTime.check();
    nowTime[5].print("程序已进行");
    nowTime[5].check();
    printf("   kmax-1-core上的kmaxtruss为 %d, 剩余的边数为 %d. \n", try_floor, numEdges_final/2); 
#endif



    delete [] edges_sub;
    delete [] edgelist_two;
    if (numEdges_sub_100 != 0) delete [] edges_sub_100;
    if (numEdges_sub_30 != 0) delete [] edges_sub_30;
    printf("kmax = %d, Edges in kmax-truss = %d.\n", try_floor, numEdges_final/2);
    calcTime.print("七、 计算时间 = [");
    allTime.print("八、 总时间 = ");

    return 0;
}
