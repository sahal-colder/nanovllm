"""Fail closed on foreign CUDA compute processes; never stop another job."""
import csv,datetime,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path

def snapshot():
    proc=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,values=line.split(':',1)
        if key in ['MemTotal','MemAvailable','SwapTotal','SwapFree']:mem[key+'_KiB']=int(values.split()[0])
    pressure=Path('/proc/pressure/memory').read_text() if Path('/proc/pressure/memory').exists() else ''
    return {'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'compute':list(csv.reader(proc.splitlines())),'gpu':gpu,'host_memory':mem,'memory_pressure':pressure}
def descendant(pid,ancestor):
    for _ in range(64):
        if pid==ancestor:return True
        try:
            status=open(f'/proc/{pid}/status').read().splitlines()
            pid=int(next(s.split()[1] for s in status if s.startswith('PPid:')))
        except (OSError,StopIteration):return False
        if pid<2:return False
    return False
def run_checked(command,stdout,stderr,telemetry):
    artifacts={};begin=time.monotonic();timeout=float(os.environ.get('NV_BENCH_TIMEOUT_SECONDS','180'))
    for index,arg in enumerate(command):
        path=Path(str(arg))
        if index==0 and not path.is_file():path=Path(shutil.which(str(arg)) or str(arg))
        if path.is_file() and (index==0 or os.access(path,os.X_OK) or path.suffix in ['.bin','.py','.sh']):
            with path.open('rb') as f:artifacts[str(path.resolve())]=hashlib.file_digest(f,'sha256').hexdigest()
    def check(ancestor=None):
        s=snapshot();s['allowed_root_pid']=ancestor;s['command']=list(map(str,command));s['artifacts_sha256']=artifacts
        process_memory={};process_cgroup={}
        for pid in {ancestor,*[int(r[0]) for r in s['compute']]}-{None}:
            try:
                lines=Path(f'/proc/{pid}/status').read_text().splitlines()
                process_memory[str(pid)]={line.split(':')[0]:line.split(':',1)[1].strip() for line in lines if line.startswith(('VmRSS:','VmSwap:','Threads:'))}
                cgroup=next(line.split(':',2)[2] for line in Path(f'/proc/{pid}/cgroup').read_text().splitlines() if line.startswith('0::'))
                directory=Path('/sys/fs/cgroup')/cgroup.lstrip('/')
                process_cgroup[str(pid)]={'path':cgroup,**{name:(directory/name).read_text().strip() for name in ['memory.current','memory.peak','memory.max','memory.swap.current','memory.swap.max','memory.events'] if (directory/name).exists()}}
            except OSError:pass
        s['process_memory']=process_memory;s['process_cgroup']=process_cgroup;telemetry.write(json.dumps(s)+'\n');telemetry.flush()
        foreign=[r for r in s['compute'] if not ancestor or not descendant(int(r[0]),ancestor)]
        if foreign:raise RuntimeError('Foreign CUDA compute process: '+repr(foreign))
        mem=s['host_memory'];reserve=int(os.environ.get('NV_BENCH_MIN_AVAILABLE_MIB','3072'))*1024
        if mem['MemAvailable_KiB']<reserve:raise RuntimeError('Host available RAM below reserved threshold; stopping own benchmark')
        if mem['SwapTotal_KiB']-mem['SwapFree_KiB']>int(os.environ.get('NV_BENCH_MAX_SWAP_MIB','512'))*1024:raise RuntimeError('Host Swap use exceeds threshold; stopping own benchmark')
        fields=s['gpu'].split(', ')
        if len(fields)>=9 and int(fields[8])-int(fields[7])<1024:raise RuntimeError('Less than 1 GiB VRAM headroom; stopping own benchmark')
        if ancestor and time.monotonic()-begin>timeout:raise TimeoutError('Benchmark exceeded bounded runtime')
    check()
    # Instrumented validation is not performance measurement; reduce host priority.
    launch=list(map(str,command))
    if Path(launch[0]).name=='compute-sanitizer':launch=['nice','-n','10',*launch]
    child=subprocess.Popen(launch,stdout=stdout,stderr=stderr,start_new_session=True)
    try:
        while child.poll() is None:
            check(child.pid);time.sleep(1)
        check()
        if child.returncode:raise subprocess.CalledProcessError(child.returncode,command)
    except BaseException:
        # Only terminate our child; retain logs and reject the entire run.
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=3)
        raise

class Phase:
    """A success marker is written only when the entire phase succeeds."""
    def __init__(self,out,name):
        self.path=out/(name+'-status.json')
        if self.path.exists():raise FileExistsError('Choose fresh phase outputs: '+str(self.path))
    def write(self,status,error=None):
        self.path.write_text(json.dumps({'status':status,'error':error,'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
    def __enter__(self):
        self.write('running');return self
    def __exit__(self,kind,value,tb):
        self.write('failed' if kind else 'complete',str(value) if kind else None)
        return False
