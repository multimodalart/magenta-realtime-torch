/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { useState, useEffect } from 'react';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import { VolumeControl } from './VolumeControl';
import { Replay } from '@mui/icons-material';

interface TransportControlsProps {
  isPlaying: boolean;
  onTogglePlay: () => void;
  volume: number;
  onVolumeChange: (v: number) => void;
  onReset: () => void;
  onResetDown?: () => void;
  onResetUp?: () => void;
  volumeSliderPosition?: 'top' | 'bottom';
  model?: string;
  resetTooltip?: string;
  showPlay?: boolean;
  showVolume?: boolean;
  isDawPlaying?: boolean;
}

function postBridge(msg: any) { (window as any).webkit?.messageHandlers?.auHost?.postMessage(msg); }

export function ZeroGpuSession() {
  const [len, setLen] = useState(120);
  const [remaining, setRemaining] = useState(120);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    const h = (e: any) => {
      const d = e.detail || {};
      if (d.len !== undefined) setLen(d.len);
      if (d.remaining !== undefined) setRemaining(d.remaining);
      if (d.playing !== undefined) setPlaying(d.playing);
    };
    window.addEventListener('mrt-gpu', h);
    return () => window.removeEventListener('mrt-gpu', h);
  }, []);
  const fmt = (s: number) => { s = Math.max(0, Math.floor(s)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); };
  return (
    <div title="ZeroGPU session length (1-4 min)" style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 12px', borderRadius: '999px', background: 'var(--color-raised, #36373a)', fontFamily: "'Google Sans Text', system-ui", fontSize: '12px', fontWeight: 500, color: 'rgba(255,255,255,0.85)', whiteSpace: 'nowrap', flexShrink: 0 }}>
      <span style={{ color: '#FFC23C', fontSize: '14px', lineHeight: 1 }}>⚡</span>
      <input type="range" min={60} max={240} step={30} value={len} disabled={playing}
        onChange={(e) => { const v = +e.target.value; setLen(v); postBridge({ type: 'gpuSession', value: v }); }}
        style={{ width: '74px', accentColor: '#FFC23C', cursor: 'pointer', opacity: playing ? 0.4 : 1 }} />
      <span style={{ minWidth: '30px', fontVariantNumeric: 'tabular-nums', color: (playing && remaining < 30) ? '#FF4C8D' : 'rgba(255,255,255,0.85)' }}>{fmt(playing ? remaining : len)}</span>
    </div>
  );
}

export function TransportControls({
  isPlaying,
  onTogglePlay,
  volume,
  onVolumeChange,
  onReset,
  onResetDown,
  onResetUp,
  volumeSliderPosition = 'top',
  model,
  resetTooltip = 'Reset model state',
  showPlay = true,
  showVolume = true,
  isDawPlaying = false,
}: TransportControlsProps) {
  const noModel = !model || model === 'No model loaded';
  const playButton = (
    <button
      onClick={noModel ? undefined : onTogglePlay}
      disabled={noModel}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: '56px',
        height: '56px',
        borderRadius: '50%',
        border: 'none',
        background: isDawPlaying ? '#FF7A00' : '#FFF',
        color: '#000',
        padding: 0,
        flexShrink: 0,
        opacity: noModel ? 0.4 : 1,
        animation: isDawPlaying ? 'magenta-pulse 2s infinite ease-in-out' : 'none',
      }}
    >
      <span className="material-icons" style={{ fontSize: '28px' }}>
        {isDawPlaying ? 'cable' : (isPlaying ? 'pause' : 'play_arrow')}
      </span>
    </button>
  );

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
    }}>
      {/* Reset */}
      <Tooltip title={resetTooltip}>
        <IconButton
          onClick={onReset}
          onMouseDown={onResetDown}
          onMouseUp={onResetUp}
          onMouseLeave={onResetUp}
          sx={{
            width: 40,
            height: 40,
          }}
        >
          <Replay sx={{ fontSize: 20 }} />
        </IconButton>
      </Tooltip>

      {/* Play/Pause — large circle */}
      {showPlay && (noModel ? (
        <Tooltip title="No model selected" placement="top">
          <span>{playButton}</span>
        </Tooltip>
      ) : isDawPlaying ? (
        <Tooltip title="Linked to DAW" placement="top">
          <span>{playButton}</span>
        </Tooltip>
      ) : (
        playButton
      ))}


      {/* Volume */}
      {showVolume && (
        <VolumeControl
          volume={volume}
          onVolumeChange={onVolumeChange}
          sliderPosition={volumeSliderPosition}
        />
      )}
      <ZeroGpuSession />
    </div>
  );
}
