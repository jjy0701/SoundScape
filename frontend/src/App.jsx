import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { Volume2, Plus, Trash2, Map, Settings, Upload, Play, Pause, X } from 'lucide-react';

function App() {
  // 1. 상태 관리
  const [room, setRoom] = useState({ w: 20, l: 15, abs: 0.2 });
  const [obstacles, setObstacles] = useState([{ id: 1, x: 8, y: 5, w: 3, l: 4 }]);
  const [speakers, setSpeakers] = useState([{ id: 1, x: 3, y: 3, angle: 0 }]);
  const [listener, setListener] = useState({ x: 15, y: 10 });
  const [totalDb, setTotalDb] = useState(-100);
  const [heatmap, setHeatmap] = useState(null);
  const [dragging, setDragging] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showSettings, setShowSettings] = useState(false); // 🔥 넷플릭스 스타일 숨김 메뉴 상태

  // 2. 오디오 엔진 Ref
  const audioCtx = useRef(null);
  const gainNode = useRef(null);
  const sourceNode = useRef(null);
  const audioBuffer = useRef(null);
  const canvasRef = useRef(null);
  const scale = 35; // 화면을 꽉 채우기 위해 스케일 업

  // --- 실시간 가청화 로직 ---
  const updateAudioGain = useCallback((db) => {
    if (gainNode.current && audioCtx.current) {
        const gainVal = Math.pow(10, (db - 10) / 20);
        gainNode.current.gain.value = Math.min(gainVal, 1.5);
    }
  }, []);

  // --- 백엔드 통신 ---
  const fetchCurrentDb = useCallback(async () => {
    try {
        const res = await axios.post('http://127.0.0.1:8000/predict', {
            room_w: room.w, room_l: room.l, list_x: listener.x, list_y: listener.y,
            abs_v: room.abs, obstacles: obstacles, speakers: speakers
        });
        setTotalDb(res.data.total_db);
        updateAudioGain(res.data.total_db);
    } catch (e) { console.error(e); }
  }, [room, obstacles, speakers, listener, updateAudioGain]);

  // --- 파일 업로드 및 재생 ---
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setAudioFile(file.name);

    if (!audioCtx.current) audioCtx.current = new (window.AudioContext || window.webkitAudioContext)();
    await audioCtx.current.resume();
    const arrayBuf = await file.arrayBuffer();
    audioBuffer.current = await audioCtx.current.decodeAudioData(arrayBuf);

    sourceNode.current = audioCtx.current.createBufferSource();
    sourceNode.current.buffer = audioBuffer.current;
    sourceNode.current.loop = true;

    gainNode.current = audioCtx.current.createGain();
    sourceNode.current.connect(gainNode.current);
    gainNode.current.connect(audioCtx.current.destination);

    sourceNode.current.start();
    setIsPlaying(true);
    
    await fetchCurrentDb();
  };

  const togglePlayPause = async () => {
    if (!audioFile || !audioBuffer.current) return;

    if (isPlaying) {
      if (sourceNode.current) sourceNode.current.stop();
      setIsPlaying(false);
    } else {
      if (!audioCtx.current) audioCtx.current = new (window.AudioContext || window.webkitAudioContext)();
      await audioCtx.current.resume();

      sourceNode.current = audioCtx.current.createBufferSource();
      sourceNode.current.buffer = audioBuffer.current;
      sourceNode.current.loop = true;

      gainNode.current = audioCtx.current.createGain();
      sourceNode.current.connect(gainNode.current);
      gainNode.current.connect(audioCtx.current.destination);

      sourceNode.current.start();
      setIsPlaying(true);
      
      await fetchCurrentDb();
    }
  };

  const clearAudio = () => {
    if (sourceNode.current && isPlaying) sourceNode.current.stop();
    setAudioFile(null);
    setIsPlaying(false);
    audioBuffer.current = null;
  };

  const generateHeatmap = async () => {
    setShowSettings(false); // 히트맵 생성 시 메뉴 닫기
    const res = await axios.post('http://127.0.0.1:8000/predict_map', {
      room_w: room.w, room_l: room.l, list_x: 0, list_y: 0,
      abs_v: room.abs, obstacles: obstacles, speakers: speakers
    });
    setHeatmap(res.data.map);
  };

  // --- 드래그 로직 ---
  const onMouseDown = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / scale;
    const y = (e.clientY - rect.top) / scale;
    if (Math.hypot(x - listener.x, y - listener.y) < 0.7) return setDragging({ type: 'L' });
    const sIdx = speakers.findIndex(s => Math.hypot(x - s.x, y - s.y) < 0.5);
    if (sIdx !== -1) return setDragging({ type: 'S', id: speakers[sIdx].id });
    const oIdx = obstacles.findIndex(o => x > o.x && x < o.x + o.w && y > o.y && y < o.y + o.l);
    if (oIdx !== -1) return setDragging({ type: 'O', id: obstacles[oIdx].id, ox: x - obstacles[oIdx].x, oy: y - obstacles[oIdx].y });
  };

  const onMouseMove = (e) => {
    if (!dragging) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(room.w, (e.clientX - rect.left) / scale));
    const y = Math.max(0, Math.min(room.l, (e.clientY - rect.top) / scale));
    if (dragging.type === 'L') setListener({ x, y });
    else if (dragging.type === 'S') setSpeakers(speakers.map(s => s.id === dragging.id ? { ...s, x, y } : s));
    else if (dragging.type === 'O') setObstacles(obstacles.map(o => o.id === dragging.id ? { ...o, x: x - dragging.ox, y: y - dragging.oy } : o));
  };

  useEffect(() => {
    if (isPlaying) fetchCurrentDb();
  }, [listener, isPlaying, fetchCurrentDb]);

  // --- 렌더링 (대표님 커스텀 캔버스 로직 유지) ---
  useEffect(() => {
    const ctx = canvasRef.current.getContext('2d');
    const w = room.w * scale;
    const h = room.l * scale;
    
    // 무대 배경 (나무 무대)
    const gradient = ctx.createLinearGradient(0, 0, w, h);
    gradient.addColorStop(0, '#3d2817');
    gradient.addColorStop(0.5, '#5a3a2a');
    gradient.addColorStop(1, '#3d2817');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, w, h);
    
    // 무대 테두리
    ctx.strokeStyle = '#d4af37';
    ctx.lineWidth = 3;
    ctx.strokeRect(0, 0, w, h);
    
    // 히트맵
    if (heatmap) {
      heatmap.forEach((row, ri) => {
        row.forEach((val, ci) => {
          const intensity = Math.max(0, Math.min(100, val + 20));
          ctx.fillStyle = `rgba(255, ${Math.floor(200 - intensity * 1.5)}, 100, ${0.3 + intensity / 300})`;
          ctx.fillRect(ci * 0.5 * scale, ri * 0.5 * scale, 0.5 * scale, 0.5 * scale);
        });
      });
    }
    
    // 장애물
    obstacles.forEach(o => {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
      ctx.fillRect(o.x * scale, o.y * scale, o.w * scale, o.l * scale);
      ctx.strokeStyle = '#d4af37';
      ctx.lineWidth = 2;
      ctx.strokeRect(o.x * scale, o.y * scale, o.w * scale, o.l * scale);
    });
    
    // 스피커
    speakers.forEach(s => {
      ctx.save();
      ctx.translate(s.x * scale, s.y * scale);
      ctx.rotate((s.angle * Math.PI) / 180);
      
      ctx.fillStyle = '#d4af37';
      ctx.beginPath();
      ctx.moveTo(-4, -8); ctx.lineTo(4, -8); ctx.lineTo(6, 8); ctx.lineTo(-6, 8);
      ctx.closePath(); ctx.fill();
      
      ctx.fillStyle = '#ffed4e';
      ctx.beginPath(); ctx.arc(0, 2, 5, 0, Math.PI * 2); ctx.fill();
      
      ctx.strokeStyle = '#ffed4e'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(0, -6); ctx.lineTo(0, -15); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(-2, -12); ctx.lineTo(0, -15); ctx.lineTo(2, -12); ctx.stroke();
      
      ctx.restore();
    });
    
    // 리스너
    ctx.fillStyle = 'rgba(212, 175, 55, 0.8)';
    ctx.beginPath(); ctx.arc(listener.x * scale, listener.y * scale, 10, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#ffed4e'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(listener.x * scale, listener.y * scale, 10, 0, Math.PI * 2); ctx.stroke();
    
    ctx.fillStyle = '#ffed4e';
    ctx.beginPath(); ctx.arc(listener.x * scale, listener.y * scale - 3, 4, 0, Math.PI * 2); ctx.fill();
  }, [room, obstacles, speakers, listener, heatmap]);

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh', background: '#0a0a0c', overflow: 'hidden', color: '#fff', fontFamily: 'sans-serif' }}>
      
      {/* 1. 배경 조명 효과 */}
      <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', background: 'radial-gradient(circle at 50% 50%, #2d2416 0%, #0a0a0c 80%)', zIndex: 0 }} />

      {/* 2. 상단 중앙 플로팅 UI (넷플릭스 스타일 dB 미터) */}
      <div style={{ position: 'absolute', top: '30px', left: '50%', transform: 'translateX(-50%)', zIndex: 10, background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(10px)', padding: '15px 40px', borderRadius: '40px', border: '1px solid #d4af37', display: 'flex', alignItems: 'center', gap: '20px', boxShadow: '0 10px 30px rgba(0,0,0,0.8)' }}>
        <Volume2 color="#d4af37" size={24} />
        <span style={{ fontSize: '26px', fontWeight: 'bold', color: '#ffed4e', letterSpacing: '2px' }}>{Number(totalDb).toFixed(1)} <small style={{fontSize:'14px', color:'#d4af37'}}>dB</small></span>
        <div style={{ width: '2px', height: '20px', background: 'rgba(212,175,55,0.3)' }} />
        <button onClick={() => setShowSettings(true)} style={{ background: 'none', border: 'none', color: '#d4af37', cursor: 'pointer', display: 'flex', alignItems: 'center', transition: 'transform 0.2s' }} onMouseOver={e => e.currentTarget.style.transform='rotate(90deg)'} onMouseOut={e => e.currentTarget.style.transform='none'}>
          <Settings size={26} />
        </button>
      </div>

      {/* 3. 메인 캔버스 영역 (풀스크린 정중앙 배치) */}
      <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 5 }}>
        <div style={{ position: 'relative', borderRadius: '12px', boxShadow: '0 20px 60px rgba(0,0,0,0.9)' }}>
          <canvas
            ref={canvasRef} width={room.w * scale} height={room.l * scale}
            onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={() => setDragging(null)} onMouseLeave={() => setDragging(null)}
            style={{ borderRadius: '12px', cursor: dragging ? 'grabbing' : 'crosshair' }}
          />
        </div>
      </div>

      {/* 4. 우측 슬라이딩 설정 패널 (Settings) */}
      <div style={{ position: 'absolute', top: 0, right: showSettings ? 0 : '-420px', width: '400px', height: '100%', background: 'rgba(26,26,26,0.95)', backdropFilter: 'blur(20px)', zIndex: 100, transition: 'right 0.4s cubic-bezier(0.4, 0, 0.2, 1)', padding: '40px', borderLeft: '2px solid #d4af37', overflowY: 'auto', boxShadow: '-10px 0 30px rgba(0,0,0,0.8)' }}>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
          <h2 style={{ color: '#d4af37', fontSize: '24px', margin: 0 }}>🎼 Stage Control</h2>
          <button onClick={() => setShowSettings(false)} style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer' }}><X size={24} /></button>
        </div>
        
        <div style={{ background: '#3d2817', padding: '20px', borderRadius: '12px', marginBottom: '25px', border: '1px solid #d4af37' }}>
            <h4 style={{ color: '#ffed4e', marginBottom: '15px', marginTop: 0 }}><Upload size={16} style={{verticalAlign:'middle'}}/> 오디오 트랙</h4>
            <input type="file" accept=".wav" onChange={handleFileUpload} style={{ width: '100%', fontSize: '12px', color: '#ccc' }} />
            {audioFile && (
              <div style={{ marginTop: '15px' }}>
                <p style={{fontSize:'13px', color:'#ffed4e', margin: '0 0 10px 0'}}>♪ {audioFile}</p>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button onClick={togglePlayPause} style={{ flex: 1, padding: '10px', background: isPlaying ? '#c41e3a' : '#228b22', color: 'white', border: '1px solid #d4af37', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '5px' }}>
                    {isPlaying ? <><Pause size={16}/> PAUSE</> : <><Play size={16}/> PLAY</>}
                  </button>
                  <button onClick={clearAudio} style={{ flex: 0.3, padding: '10px', background: '#444', color: 'white', border: '1px solid #d4af37', borderRadius: '6px', cursor: 'pointer' }}>
                    <Trash2 size={16} style={{margin:'0 auto'}}/>
                  </button>
                </div>
              </div>
            )}
        </div>

        <section style={{ marginBottom: '25px' }}>
          <h4 style={{ color: '#d4af37', marginBottom: '10px' }}>🏛️ 무대 규격</h4>
          <label style={{ fontSize: '12px', color: '#ccc' }}>가로: {room.w}m</label>
          <input type="range" min="10" max="30" value={room.w} onChange={e => setRoom({...room, w: +e.target.value})} style={{width:'100%', accentColor: '#d4af37'}}/>
          <label style={{ fontSize: '12px', color: '#ccc', display: 'block', marginTop: '10px' }}>세로: {room.l}m</label>
          <input type="range" min="8" max="25" value={room.l} onChange={e => setRoom({...room, l: +e.target.value})} style={{width:'100%', accentColor: '#d4af37'}}/>
        </section>

        <section style={{ marginBottom: '25px' }}>
          <h4 style={{ color: '#d4af37', marginBottom: '10px' }}>🎤 스피커 방향 (첫번째)</h4>
          <input type="range" min="0" max="360" step="5" value={speakers[0]?.angle || 0} onChange={e => setSpeakers(speakers.map((s,i)=> i===0?{...s, angle:+e.target.value}:s))} style={{width:'100%', accentColor: '#d4af37'}}/>
        </section>

        <section style={{ marginBottom: '25px' }}>
          <h4 style={{ color: '#d4af37', marginBottom: '10px' }}>📦 소품 폭 조절 (첫번째)</h4>
          <input type="range" min="1" max="10" step="0.1" value={obstacles[0]?.w || 2} onChange={e => setObstacles(obstacles.map((o,i)=> i===0?{...o, w:+e.target.value}:o))} style={{width:'100%', accentColor: '#d4af37'}}/>
        </section>

        <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
            <button onClick={() => setSpeakers([...speakers, { id: Date.now(), x: 2, y: 2, angle: 0 }])} style={{ flex: 1, padding: '12px', background: 'transparent', color: '#d4af37', border: '1px solid #d4af37', borderRadius: '6px', cursor: 'pointer', transition: 'background 0.2s' }} onMouseOver={e=>e.currentTarget.style.background='rgba(212,175,55,0.1)'} onMouseOut={e=>e.currentTarget.style.background='transparent'}>+ 스피커</button>
            <button onClick={() => setObstacles([...obstacles, { id: Date.now(), x: 5, y: 5, w: 2, l: 2 }])} style={{ flex: 1, padding: '12px', background: 'transparent', color: '#d4af37', border: '1px solid #d4af37', borderRadius: '6px', cursor: 'pointer', transition: 'background 0.2s' }} onMouseOver={e=>e.currentTarget.style.background='rgba(212,175,55,0.1)'} onMouseOut={e=>e.currentTarget.style.background='transparent'}>+ 소품</button>
        </div>

        <button onClick={generateHeatmap} style={{ width: '100%', padding: '15px', background: '#d4af37', color: '#1a1a1a', border: 'none', borderRadius: '6px', fontWeight: 'bold', fontSize: '16px', cursor: 'pointer' }}>
          <Map size={18} style={{verticalAlign:'middle', marginRight:'8px'}}/> 히트맵 렌더링
        </button>

      </div>
    </div>
  );
}
export default App;