#!/usr/bin/env python3
# ============================================================
# TRADING AUTOMATION: ORO/DXY Analysis with Claude AI
# ============================================================
# Version: 1.0.0
# Author: Fabio (FaZuCa Consulting)
# License: MIT
# 
# Description:
#   Automated analysis of XAUUSD (Gold) and DXY trading setups
#   using Claude AI. Validates entries against professional
#   trading rules (macro bias, inverse correlation, liquidity
#   sweeps, volume confirmation).
#
# Usage:
#   python nivel2b_trading_analysis.py --data data.csv
#   python nivel2b_trading_analysis.py --simulate
#
# ============================================================

import anthropic
import json
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = "claude-opus-4-1-20250805"
MAX_TOKENS = 1024
ANALYSIS_OUTPUT_DIR = Path("analysis_results")

# ============================================================
# SIMULATED DATA (for testing without real market data)
# ============================================================

SIMULATED_ORO_DATA = {
    "par": "XAUUSD",
    "timeframe": "15M",
    "velas": [
        {"bar": 50, "open": 2520.00, "high": 2522.50, "low": 2519.00, "close": 2521.00, "volume": 1200},
        {"bar": 49, "open": 2519.00, "high": 2523.00, "low": 2518.50, "close": 2520.50, "volume": 1400},
        {"bar": 48, "open": 2518.00, "high": 2521.00, "low": 2517.00, "close": 2519.50, "volume": 1100},
        {"bar": 47, "open": 2522.00, "high": 2525.00, "low": 2520.00, "close": 2523.50, "volume": 1600},
        {"bar": 46, "open": 2520.50, "high": 2524.00, "low": 2519.50, "close": 2521.00, "volume": 1300},
    ],
    "pdh": 2530.00,
    "pdl": 2518.00,
    "tdh": 2527.00,
    "tdl": 2519.00,
    "rsi_15m": 62,
    "ema200_15m": 2520.00,
    "sesgo_1d": "BULL",
    "sesgo_4h": "BULL",
}

SIMULATED_DXY_DATA = {
    "par": "DXY",
    "timeframe": "15M",
    "velas": [
        {"bar": 50, "open": 104.25, "high": 104.35, "low": 104.15, "close": 104.30, "volume": 2100},
        {"bar": 49, "open": 104.20, "high": 104.40, "low": 104.10, "close": 104.25, "volume": 2300},
        {"bar": 48, "open": 104.30, "high": 104.45, "low": 104.20, "close": 104.35, "volume": 2500},
        {"bar": 47, "open": 104.35, "high": 104.50, "low": 104.25, "close": 104.40, "volume": 2700},
        {"bar": 46, "open": 104.40, "high": 104.55, "low": 104.30, "close": 104.45, "volume": 2900},
    ],
    "pdh": 104.75,
    "pdl": 104.10,
    "tdh": 104.80,
    "tdl": 104.15,
    "rsi_15m": 75,
    "ema200_15m": 104.20,
    "sesgo_1d": "BEAR",
    "sesgo_4h": "BEAR",
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_api_key() -> str:
    """Retrieve API key from environment variable."""
    import os
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "❌ ERROR: ANTHROPIC_API_KEY environment variable not set.\n"
            "Set it with: $env:ANTHROPIC_API_KEY='your-key-here' (PowerShell)\n"
            "or: export ANTHROPIC_API_KEY='your-key-here' (Linux/Mac)"
        )
    return api_key

def build_analysis_prompt(oro_data: Dict[str, Any], dxy_data: Dict[str, Any]) -> str:
    """Build the Claude prompt for trading analysis."""
    prompt = f"""
Eres un analista de trading EXPERTO en ORO (XAUUSD) y su correlación con DXY.

Tu checklist de entrada es:
1. **Sesgo macro** (1D OK)
2. **Correlación inversa** (ORO en resistencia + DXY en soporte, o viceversa)
3. **Liquidity sweep confirmado** 
4. **FVG retroceso** (zona de compra/venta)
5. **Confirmación DXY primero** (5M o 15M)

DATOS ACTUALES:

**ORO (XAUUSD) - 15M:**
- Últimas velas: {json.dumps(oro_data['velas'][:3], indent=2)}
- PDH: {oro_data['pdh']} | PDL: {oro_data['pdl']}
- TDH: {oro_data['tdh']} | TDL: {oro_data['tdl']}
- RSI 15M: {oro_data['rsi_15m']}
- EMA200 15M: {oro_data['ema200_15m']}
- Sesgo 1D: {oro_data['sesgo_1d']} | Sesgo 4H: {oro_data['sesgo_4h']}

**DXY - 15M:**
- Últimas velas: {json.dumps(dxy_data['velas'][:3], indent=2)}
- PDH: {dxy_data['pdh']} | PDL: {dxy_data['pdl']}
- TDH: {dxy_data['tdh']} | TDL: {dxy_data['tdl']}
- RSI 15M: {dxy_data['rsi_15m']}
- EMA200 15M: {dxy_data['ema200_15m']}
- Sesgo 1D: {dxy_data['sesgo_1d']} | Sesgo 4H: {dxy_data['sesgo_4h']}

ANÁLISIS REQUERIDO:

1. ¿Se cumple la correlación inversa? (ORO vs DXY posiciones opuestas)
2. ¿DXY confirmó con volumen? (últimas velas muestran confirmación)
3. ¿Hay liquidity sweep válido en ORO?
4. ¿Hay FVG retroceso en ORO?
5. ¿El sesgo macro permite LONG o SHORT?

RESPONDE EN ESTE FORMATO JSON ÚNICAMENTE:

{{
    "correlacion_inversa": true/false,
    "dxy_confirmado": true/false,
    "liquidity_sweep": true/false,
    "fvg_retroceso": true/false,
    "sesgo_macro": "LONG" / "SHORT" / "NEUTRAL",
    "setup_valido": true/false,
    "entrada": "BUY" / "SELL" / "NO ENTRY",
    "entrada_precio": NUMBER,
    "stop_loss": NUMBER,
    "take_profit": NUMBER,
    "riesgo_recompensa": "1:X.X",
    "probabilidad": NUMBER (0-100),
    "razon": "Explicación breve del setup"
}}

Sé PRECISO y CRÍTICO. Solo da ENTRADA si se cumplen TODOS los criterios.
"""
    return prompt

def parse_claude_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from Claude response."""
    try:
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            return None
        json_str = response_text[json_start:json_end]
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def save_analysis_results(analysis: Dict[str, Any], filename: str = None) -> Path:
    """Save analysis results to JSON file."""
    ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True)
    
    if not filename:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"analysis_{timestamp}.json"
    
    filepath = ANALYSIS_OUTPUT_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    return filepath

def display_results(setup: Dict[str, Any], tokens_used: tuple) -> None:
    """Display formatted analysis results."""
    input_tokens, output_tokens = tokens_used
    total_tokens = input_tokens + output_tokens
    
    print("\n✅ SETUP ANALIZADO:")
    print("=" * 70)
    print(f"  Entrada: {setup.get('entrada', 'NO ENTRY')}")
    print(f"  Precio entrada: {setup.get('entrada_precio', 'N/A')}")
    print(f"  Stop Loss: {setup.get('stop_loss', 'N/A')}")
    print(f"  Take Profit: {setup.get('take_profit', 'N/A')}")
    print(f"  R:R: {setup.get('riesgo_recompensa', 'N/A')}")
    print(f"  Probabilidad: {setup.get('probabilidad', 'N/A')}%")
    print(f"  Correlación inversa: {'✅' if setup.get('correlacion_inversa') else '❌'}")
    print(f"  DXY confirmado: {'✅' if setup.get('dxy_confirmado') else '❌'}")
    print(f"  Liquidity sweep: {'✅' if setup.get('liquidity_sweep') else '❌'}")
    print(f"  FVG retroceso: {'✅' if setup.get('fvg_retroceso') else '❌'}")
    print(f"\n  Razón: {setup.get('razon', 'N/A')}")
    print("=" * 70)
    
    print(f"\n📊 ESTADÍSTICAS:")
    print(f"  Input tokens: {input_tokens}")
    print(f"  Output tokens: {output_tokens}")
    print(f"  Total: {total_tokens}")
    cost = (total_tokens * 0.0001) / 1000
    print(f"  Costo aproximado: ${cost:.4f}")

# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_setup(use_simulated: bool = False, data_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Main function to analyze ORO/DXY trading setup with Claude.
    
    Args:
        use_simulated: Use simulated data instead of real data
        data_file: Path to CSV file with real data
    
    Returns:
        Dictionary with analysis results
    """
    
    # Get API key
    try:
        api_key = get_api_key()
    except EnvironmentError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Use simulated data
    oro_data = SIMULATED_ORO_DATA.copy()
    dxy_data = SIMULATED_DXY_DATA.copy()
    
    # Build analysis prompt
    prompt = build_analysis_prompt(oro_data, dxy_data)
    
    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Display status
    print("🚀 Analizando setup ORO/DXY con Claude...")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Call Claude API
    try:
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
    except anthropic.APIError as e:
        print(f"❌ API Error: {e}")
        sys.exit(1)
    
    # Parse response
    respuesta_text = response.content[0].text
    
    print("\n📊 RESPUESTA DE CLAUDE:")
    print("=" * 70)
    print(respuesta_text)
    print("=" * 70)
    
    # Parse JSON
    setup = parse_claude_response(respuesta_text)
    
    if setup:
        # Display results
        display_results(setup, (response.usage.input_tokens, response.usage.output_tokens))
        
        # Save results
        filepath = save_analysis_results({
            "timestamp": datetime.now().isoformat(),
            "setup": setup,
            "tokens": {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
                "total": response.usage.input_tokens + response.usage.output_tokens
            },
            "oro_data": oro_data,
            "dxy_data": dxy_data
        })
        
        print(f"\n💾 Análisis guardado en: {filepath}")
    else:
        print("\n⚠️ No se pudo parsear JSON de la respuesta")
        print("Claude respondió en formato de texto (válido pero no estructurado)")
    
    return setup if setup else {}

# ============================================================
# CLI INTERFACE
# ============================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Trading automation: ORO/DXY analysis with Claude AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nivel2b_trading_analysis.py --simulate
  python nivel2b_trading_analysis.py --data market_data.csv
  python nivel2b_trading_analysis.py
        """
    )
    
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use simulated data for testing"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        help="Path to CSV file with market data"
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyze_setup(use_simulated=args.simulate or not args.data, data_file=args.data)

if __name__ == "__main__":
    main()
