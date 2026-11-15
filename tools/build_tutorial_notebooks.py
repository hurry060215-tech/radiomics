"""Generate stepwise notebooks from the reviewed executable examples."""
import ast
import json
from pathlib import Path
import textwrap
import uuid

ROOT=Path(__file__).resolve().parents[1]
TITLES=['NIfTI 与 ROI 基础提取','影像与临床二分类','多分类与连续值回归','多区域、Delta 与生境','病理 patch 到患者建模','Embedding 对齐与局部覆盖图']
TOPICS={
 'extracted':('准备影像与队列','生成小型合成 NIfTI 和 ROI，提取特征。先看样本 ID 与患者 ID 的对应关系；同一患者的多个病灶必须留在同一划分。'),
 'clinical':('整理临床数据','临床表每位患者一行，随后按患者 ID 关联到病灶。many_to_one 校验用于防止重复连接。'),
 'data':('检查输入表','检查特征列、结局列和 train/test/external 队列。结局与 ID 不进入特征矩阵；测试集不参与填补、缩放或筛选拟合。'),
 'selector':('配置特征筛选','这里关闭单因素和稀疏筛选以展示稳定的完整流程。真实研究应事先声明筛选方法和阈值；筛选在训练折内拟合。'),
 'selection':('配置任务对应筛选','多分类不能把类别编码直接作为连续回归目标。选择器随整个 Pipeline 在患者分组交叉验证内拟合。'),
 'image':('拟合影像模型','指定原生 Rust 后端及患者层级。外层 CV 用于评价，内层 CV 用于选择配置；关注实际患者集合而不只是 fold 编号。'),
 'clinic':('使用共享划分拟合临床模型','复用影像模型的 SplitPlan，确保两条分支的训练/验证患者完全一致。分类变量必须显式声明。'),
 'fusion':('用 OOF 概率进行融合','融合拟合使用基础模型对未参与该折训练患者的预测。融合自身的训练表现属于表观表现，不能代替独立测试。'),
 'result':('执行原生分析并检查结果','阅读本单元的核心调用与参数。模型训练只使用训练队列；评价时显式保留患者单位和队列身份。'),
 'external':('无标签外部预测','移除结局后进行预测，确认保存的填补、缩放、特征与类别顺序被复用。禁止重新拟合外部数据。'),
 'restored':('保存与重新加载','使用类型化 .rad 模型，检查加载后预测一致。重新运行保存单元前请改用新的输出目录。'),
 'report':('阅读任务评价','二分类关注 AUC 与校准，多分类关注分类别及 macro/micro/weighted 汇总，回归关注 MAE/RMSE/R²。合成数据分数不是临床证据。'),
 'tables':('严格对齐区域和模态','按患者、病灶、时间点对齐；模态与区域是显式可变化的字段。输入影像已经对齐，本包没有执行配准。'),
 'merged':('拼接多区域特征','保留来源前缀和配对证据，检查结果中的 ID、列名与参数。'),
 'change':('计算纵向变化','显式指定基线与随访。相对变化是比例；零分母示例使用缺失策略，不伪造零变化。'),
 'training':('已有矩阵上的局部生境','输入是已有强度/特征矩阵。聚类标签仅对本次拟合有意义，不能解释成跨患者一致的组织类别。'),
 'habitat':('拟合与保存聚类','保存缩放参数和中心；预测时复用这些状态，并检查模型加载后结果一致。'),
 'table':('按 tile ID 绑定特征','输入必须明确 tile ID、患者/切片层级及坐标。行数相同不能证明 embedding 对齐。'),
 'aggregated':('从 tile 聚合到患者','默认先 tile→slide，再切片等权→patient。全被排除的切片保留空状态，不生成零向量。'),
 'aggregate':('从 tile 聚合到患者','默认先 tile→slide，再切片等权→patient。查看患者层特征名和身份信息。'),
 'patients':('准备患者级建模表','用患者 ID 做一对一结局连接；明确处理无结局或空结果，不静默补零。'),
 'patient':('训练患者级模型','患者级模型输出只代表患者，不能广播到局部病理图。'),
 'tiles':('准备真实 tile 级模型','tile 模型保留每个 tile 的样本身份，但 CV 仍按患者分组。'),
 'tile':('拟合 tile 模型','局部概率图来自实际 tile 级预测；患者内多个 tile 不得跨训练/验证集合。'),
 'local':('生成局部分数','保留 tile、slide、patient 层级及预测来源；随后按真实坐标回填。'),
 'canvas':('定义覆盖图坐标','画布使用 level-0 像素边界，x 向右、y 向下。覆盖计数为零的位置透明；重叠部分取平均。'),
 'objects':('已有分割 mask 统计','区分 instance 与 semantic mask。这里的对象数是 patch 级；没有全局实例 ID 时不能把重叠 patch 计数相加。'),
 'config':('显式提取参数','Original 特征，固定分箱参数和方向聚合方式。示例参数不代表所有 CT/MR/PET 的通用推荐。'),
 'cohort':('导出并读取队列清单','清单关联样本、患者、影像与 ROI。路径相对 CSV 解析，检查每条记录是否对应正确文件。'),
}

def cell(kind,source):
 d=dict(cell_type=kind,id=uuid.uuid4().hex[:12],metadata={},source=source)
 if kind=='code':d.update(execution_count=None,outputs=[])
 return d

for number,path in enumerate(sorted((ROOT/'examples/native').glob('[0-9][0-9]_*.py'))):
 tree=ast.parse(path.read_text(encoding='utf-8'))
 main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 setup='''from pathlib import Path
import sys, tempfile
from IPython.display import display, Image
# 从 Notebook 所在目录或仓库根目录启动均可。
REPO = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'examples/native/_shared.py').exists()), None)
if REPO is None:
    raise RuntimeError('请先下载完整仓库，并在仓库内启动 Jupyter。')
sys.path.insert(0, str(REPO/'examples/native'))
import ov_radiomics as rad
print(rad.backend_info())
# 每次完整运行使用新目录，避免覆盖已有结果。
output = Path(tempfile.mkdtemp(prefix='radiomics-'''+path.stem+'''-'))
root = output
print('输出目录:', output)
'''
 imports='\n'.join(ast.unparse(n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)))
 helpers='\n\n'.join(ast.unparse(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name!='main')
 cells=[cell('markdown',f'# {TITLES[number]}\n\n按 **Shift+Enter** 从上到下逐单元运行。变量保留在当前内核中；如果修改上游参数，请重新运行其后的单元。\n\n本教程使用固定合成数据，演示软件流程，不提供临床效能结论。需要完整仓库和对应原生安装包；源码环境为 `45140bf` 构建。'),cell('markdown','## 1. 环境与独立输出目录\n\n确认后端为 Rust。后续每步都可以检查中间对象；示例数据生成辅助函数只负责小型输入构造。'),cell('code',setup+'\n'+imports+'\n'+helpers)]
 nodes=[]
 for n in main.body:
  code=ast.unparse(n)
  if number==0 and ('parser' in code.split('\n')[0] or code.startswith('args =') or code.startswith('root =') or code.startswith('root.mkdir') or code.startswith('if args.cli:')):continue
  if code.startswith('output = output_arguments'):continue
  if number==2 and isinstance(n,ast.For):
   for task in ['multiclass','regression']:
    nodes.append(ast.Assign(targets=[ast.Name(id='task',ctx=ast.Store())],value=ast.Constant(task)))
    nodes.extend(n.body)
  else:nodes.append(n)
 groups=[];group=[];topic=None
 for n in nodes:
  key=n.targets[0].id if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) else None
  if key in TOPICS or len(group)>=5:
   if group:groups.append((topic,group))
   group=[];topic=key
  group.append(n)
 if group:groups.append((topic,group))
 for index,(topic,group) in enumerate(groups,2):
  title,explanation=TOPICS.get(topic,('准备数据与保存中间结果','运行并检查本步的数据构造、校验或导出操作。遇到报错时先检查上一单元的输出和对象，而不要跳过校验。'))
  cells.append(cell('markdown',f'## {index}. {title}\n\n{explanation}'))
  code='\n'.join(ast.unparse(ast.fix_missing_locations(n)) for n in group)
  if topic in ('data','clinical','patients','tiles','training','report'):
   code+=f"\nif hasattr({topic}, 'head'):\n    display({topic}.head())"
  if topic in ('merged','change'):
   code+=f'\ndisplay({topic}.features.head())'
  cells.append(cell('code',code))
 cells.extend([cell('markdown','## 查看导出的图与文件\n\n查看图表的坐标、单位和预测层级。输出目录保留完整结果；这里显示前两幅图。'),cell('code',"for figure in sorted(output.glob('*.png'))[:2]:\n    display(Image(filename=str(figure)))\nprint('结果文件:', [p.name for p in sorted(output.iterdir())])"),cell('markdown','## 换成自己的数据\n\n保持示例的 ID、患者/切片层级与列角色约定，先替换数据读取步骤，再重新运行下游单元。先确认几何、单位、缺失状态和队列重叠；不要用测试集决定阈值或筛选参数。\n\n重新运行全部单元会新建结果目录；只重跑保存单元可能因为文件已存在而报错，这是显式防覆盖行为。')])
 notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),cells=cells)
 dest=ROOT/'website/tutorials'/(path.stem+'.ipynb');dest.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
 # Notebook is now the page source; remove the superseded Markdown wrapper.
 wrapper=dest.with_suffix('.md')
 if wrapper.exists():wrapper.unlink()
 print(path.stem,len([c for c in cells if c['cell_type']=='code']),'code cells')
conf=ROOT/'website/conf.py';conf.write_text(conf.read_text(encoding='utf-8').replace("exclude_patterns += ['tutorials/*.ipynb']",''),encoding='utf-8')
