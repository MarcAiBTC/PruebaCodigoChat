"""
app.py
------

Streamlit application for the financial portfolio manager.  This app
implements user authentication, secure session handling, portfolio
management (uploading, creating, listing and downloading), and rich
visualisations of portfolio performance.  Each user has an isolated
workspace where their data is stored and analysed.  The application
supports multiple asset classes such as stocks, ETFs, precious metals
and cryptocurrencies via real‑time pricing using yfinance.

To run this application locally:

    streamlit run app.py

Make sure to install the required dependencies first:

    pip install streamlit pandas numpy yfinance plotly

"""

import os
from datetime import datetime
from typing import Optional
import time

import pandas as pd
import numpy as np
import streamlit as st

from auth import authenticate_user, register_user
import portfolio_utils as putils


# -----------------------------------------------------------------------------
# Configuration and Setup
# -----------------------------------------------------------------------------

# Page configuration
st.set_page_config(
    page_title="Gestor de Cartera Financiera",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #e6f3ff;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .welcome-banner {
        background: linear-gradient(90deg, #e6f3ff 0%, #f0f8ff 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
    .sidebar-info {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session state initialisation
# -----------------------------------------------------------------------------

def initialize_session_state():
    """Initialize all session state variables with proper defaults."""
    defaults = {
        'authenticated': False,
        'username': '',
        'portfolio_df': None,
        'selected_portfolio_file': None,
        'price_cache': {},
        'price_cache_time': 0,
        'first_login': True,
        'portfolio_modified': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()


# -----------------------------------------------------------------------------
# Helper functions for the Streamlit UI
# -----------------------------------------------------------------------------

def show_welcome_message():
    """Display welcome message for new users."""
    if st.session_state.first_login and st.session_state.authenticated:
        st.markdown(f"""
        <div class="welcome-banner">
            <h2>¡Bienvenido/a, {st.session_state.username}! 👋</h2>
            <p>Has iniciado sesión correctamente en tu Gestor de Cartera Financiera. Aquí podrás:</p>
            <ul>
                <li>📊 <strong>Ver el resumen</strong> de tu cartera con métricas en tiempo real</li>
                <li>➕ <strong>Añadir activos</strong> manualmente o subir archivos CSV/JSON</li>
                <li>📈 <strong>Analizar el rendimiento</strong> con gráficos interactivos</li>
                <li>📚 <strong>Gestionar el historial</strong> de tus carteras guardadas</li>
            </ul>
            <p><em>💡 Consejo: Comienza añadiendo algunos activos o subiendo una cartera existente.</em></p>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.first_login = False


def load_and_set_portfolio(username: str, filename: Optional[str] = None) -> None:
    """Load a portfolio from disk and store it in session state."""
    try:
        df = putils.load_portfolio(username, filename)
        st.session_state.portfolio_df = df
        st.session_state.selected_portfolio_file = filename
        st.session_state.portfolio_modified = False
        
        if df is not None and not df.empty:
            st.success(f"✅ Cartera cargada correctamente ({len(df)} activos)")
        elif filename:
            st.warning(f"⚠️ No se pudo cargar la cartera: {filename}")
    except Exception as e:
        st.error(f"❌ Error al cargar la cartera: {str(e)}")
        st.session_state.portfolio_df = None


def get_cached_prices(tickers: list, cache_duration_minutes: int = 5) -> dict:
    """Get prices with caching to avoid excessive API calls."""
    current_time = time.time()
    cache_key = ','.join(sorted(tickers))
    
    # Check if cache is still valid
    if (cache_key in st.session_state.price_cache and 
        current_time - st.session_state.price_cache_time < cache_duration_minutes * 60):
        return st.session_state.price_cache[cache_key]
    
    # Fetch new prices
    with st.spinner("📡 Obteniendo precios actuales..."):
        prices = putils.fetch_current_prices(tickers)
    
    # Update cache
    st.session_state.price_cache[cache_key] = prices
    st.session_state.price_cache_time = current_time
    
    return prices


def validate_ticker(ticker: str) -> tuple[bool, str]:
    """Validate ticker symbol and return validation result."""
    if not ticker or len(ticker.strip()) == 0:
        return False, "El ticker no puede estar vacío"
    
    ticker = ticker.strip().upper()
    
    # Basic format validation
    if len(ticker) > 10:
        return False, "El ticker es demasiado largo (máximo 10 caracteres)"
    
    if not ticker.replace('.', '').replace('-', '').isalnum():
        return False, "El ticker contiene caracteres no válidos"
    
    return True, ""


def display_portfolio_overview() -> None:
    """
    Render the main portfolio overview page with improved error handling
    and user experience.
    """
    username = st.session_state.username
    
    st.markdown('<h1 class="main-header">📊 Resumen de Cartera</h1>', unsafe_allow_html=True)
    
    # Portfolio selection section
    st.subheader("🗂️ Seleccionar Cartera")
    portfolios = putils.list_portfolios(username)
    
    if portfolios:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            default_index = 0
            if st.session_state.selected_portfolio_file in portfolios:
                try:
                    default_index = portfolios.index(st.session_state.selected_portfolio_file)
                except ValueError:
                    default_index = 0
            
            selected_file = st.selectbox(
                "Selecciona una cartera para cargar:",
                portfolios,
                index=default_index,
                key="portfolio_select_box",
                help="Selecciona una de tus carteras guardadas"
            )
            
            if selected_file != st.session_state.selected_portfolio_file:
                load_and_set_portfolio(username, selected_file)
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            if st.button("🔄 Actualizar Lista", help="Recargar la lista de carteras"):
                st.rerun()
    else:
        st.info("📝 Aún no tienes ninguna cartera guardada. Puedes:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("• **Subir una cartera** desde la barra lateral")
        with col2:
            st.markdown("• **Añadir activos** manualmente")

    df = st.session_state.portfolio_df
    if df is None or df.empty:
        if not portfolios:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 0.5rem; margin: 2rem 0;">
                <h3>🚀 ¡Empecemos!</h3>
                <p>Para comenzar, añade algunos activos a tu cartera o sube un archivo CSV/JSON.</p>
            </div>
            """, unsafe_allow_html=True)
        return

    # Fetch current prices with caching
    tickers = df['Ticker'].tolist()
    try:
        price_dict = get_cached_prices(tickers)
        metrics_df = putils.compute_metrics(df, price_dict)
        
        # Check for pricing errors
        failed_tickers = [t for t, p in price_dict.items() if pd.isna(p)]
        if failed_tickers:
            st.warning(f"⚠️ No se pudieron obtener precios para: {', '.join(failed_tickers)}")
            
    except Exception as e:
        st.error(f"❌ Error al obtener precios: {str(e)}")
        return

    # Compute technical indicators
    with st.spinner("📊 Calculando indicadores técnicos..."):
        rsi_list = []
        vol_list = []
        
        for ticker in metrics_df['Ticker']:
            try:
                rsi, vol = putils.compute_technical_indicators(ticker)
                rsi_list.append(rsi)
                vol_list.append(vol)
            except Exception:
                rsi_list.append(np.nan)
                vol_list.append(np.nan)
        
        metrics_df['RSI'] = rsi_list
        metrics_df['Volatilidad (%)'] = vol_list

    # Portfolio summary metrics
    st.subheader("📈 Resumen Ejecutivo")
    
    total_value = metrics_df['Total Value'].sum()
    total_cost = metrics_df['Cost Basis'].sum()
    total_pl = total_value - total_cost
    total_pl_pct = (total_pl / total_cost * 100.0) if total_cost > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Valor Total",
            f"${total_value:,.2f}",
            help="Valor actual total de la cartera"
        )
    
    with col2:
        delta_color = "normal" if total_pl >= 0 else "inverse"
        st.metric(
            "📊 P/L Total",
            f"${total_pl:,.2f}",
            f"{total_pl_pct:.2f}%",
            delta_color=delta_color,
            help="Ganancia o pérdida total"
        )
    
    with col3:
        st.metric(
            "🎯 Núm. Activos",
            len(metrics_df),
            help="Número total de activos en cartera"
        )
    
    with col4:
        avg_pl = metrics_df['P/L %'].mean() if not metrics_df['P/L %'].isna().all() else 0
        st.metric(
            "📈 Rendimiento Promedio",
            f"{avg_pl:.2f}%",
            help="Rendimiento promedio de todos los activos"
        )

    # Assets detail table
    st.subheader("📋 Detalle de Activos")
    
    # Format display dataframe
    display_df = metrics_df.copy()
    format_columns = {
        'Purchase Price': lambda x: f'${x:,.2f}',
        'Current Price': lambda x: f'${x:,.2f}' if pd.notna(x) else 'N/A',
        'Total Value': lambda x: f'${x:,.2f}' if pd.notna(x) else 'N/A',
        'P/L': lambda x: f'${x:,.2f}' if pd.notna(x) else 'N/A',
        'P/L %': lambda x: f'{x:.2f}%' if pd.notna(x) else 'N/A',
        'Weight %': lambda x: f'{x:.2f}%' if pd.notna(x) else 'N/A',
        'RSI': lambda x: f'{x:.1f}' if pd.notna(x) else 'N/A',
        'Volatilidad (%)': lambda x: f'{x:.1f}%' if pd.notna(x) else 'N/A'
    }
    
    for col, formatter in format_columns.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(formatter)
    
    # Color-code the dataframe
    def highlight_pl(val):
        if 'N/A' in str(val):
            return 'color: gray'
        try:
            if '%' in str(val):
                num_val = float(str(val).replace('%', '').replace('$', '').replace(',', ''))
                if num_val > 5:
                    return 'background-color: #d4edda; color: #155724'
                elif num_val < -5:
                    return 'background-color: #f8d7da; color: #721c24'
            return ''
        except:
            return ''
    
    styled_df = display_df.style.applymap(highlight_pl, subset=['P/L %'])
    st.dataframe(styled_df, use_container_width=True, height=400)

    # Portfolio analysis section
    col1, col2 = st.columns(2)
    
    with col1:
        # Top and worst performers
        st.subheader("🏆 Mejores y Peores")
        top_df, worst_df = putils.top_and_worst_assets(metrics_df)
        
        if not top_df.empty:
            st.write("**💚 Mejores Rendimientos:**")
            top_display = top_df[['Ticker', 'P/L %', 'P/L']].head(3)
            st.dataframe(top_display, use_container_width=True, hide_index=True)
        
        if not worst_df.empty:
            st.write("**🔴 Peores Rendimientos:**")
            worst_display = worst_df[['Ticker', 'P/L %', 'P/L']].head(3)
            st.dataframe(worst_display, use_container_width=True, hide_index=True)
    
    with col2:
        # Asset allocation pie chart
        st.subheader("🥧 Distribución por Tipo")
        breakdown_df = putils.asset_breakdown(metrics_df)
        
        if not breakdown_df.empty and putils.px is not None:
            fig = putils.px.pie(
                breakdown_df,
                names='Asset Type',
                values='Total Value',
                title='Distribución por Tipo de Activo',
                color_discrete_sequence=putils.px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    # Diversification analysis
    suggestion = putils.suggest_diversification(metrics_df)
    if suggestion:
        st.warning(f"💡 **Sugerencia de Diversificación:** {suggestion}")

    # Download section
    with st.expander("💾 Descargar Datos de Cartera"):
        st.write("Descarga tu cartera actual en diferentes formatos:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = metrics_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Descargar CSV",
                data=csv_data,
                file_name=f"{username}_cartera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
                help="Descargar como archivo CSV"
            )
        
        with col2:
            json_data = metrics_df.to_json(orient="records", indent=2).encode('utf-8')
            st.download_button(
                label="📋 Descargar JSON",
                data=json_data,
                file_name=f"{username}_cartera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime='application/json',
                help="Descargar como archivo JSON"
            )


def add_asset_page() -> None:
    """
    Render the page that allows the user to add a new asset to the current
    portfolio with improved validation and user experience.
    """
    st.markdown('<h1 class="main-header">➕ Añadir Activo</h1>', unsafe_allow_html=True)
    
    username = st.session_state.username
    df = st.session_state.portfolio_df
    
    st.write("Añade un nuevo activo a tu cartera especificando el ticker, precio de compra y cantidad.")
    
    with st.form("add_asset_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            ticker = st.text_input(
                "🎯 Ticker/Símbolo",
                max_chars=10,
                help="Ej: AAPL, TSLA, BTC-USD, ^GSPC",
                placeholder="AAPL"
            ).strip().upper()
            
            purchase_price = st.number_input(
                "💰 Precio de Compra ($)",
                min_value=0.0,
                format="%.4f",
                step=0.01,
                help="Precio al que compraste el activo"
            )
        
        with col2:
            quantity = st.number_input(
                "📦 Cantidad",
                min_value=0.0,
                format="%.6f",
                step=0.001,
                help="Número de acciones/unidades"
            )
            
            asset_type = st.selectbox(
                "📊 Tipo de Activo",
                ["Acción", "ETF", "Cripto", "Oro", "Bono", "REIT", "Otro"],
                index=0,
                help="Categoría del activo para análisis de diversificación"
            )
        
        # Preview section
        if ticker and purchase_price > 0 and quantity > 0:
            cost_basis = purchase_price * quantity
            st.info(f"💡 **Vista previa:** {quantity} unidades de {ticker} con costo total de ${cost_basis:,.2f}")
        
        submitted = st.form_submit_button("➕ Agregar a Cartera", type="primary")
        
        if submitted:
            # Validation
            is_valid, error_msg = validate_ticker(ticker)
            
            if not is_valid:
                st.error(f"❌ {error_msg}")
            elif quantity <= 0:
                st.error("❌ La cantidad debe ser mayor que cero")
            elif purchase_price <= 0:
                st.error("❌ El precio de compra debe ser mayor que cero")
            else:
                # Check if ticker already exists
                existing_tickers = df['Ticker'].tolist() if df is not None and not df.empty else []
                
                if ticker in existing_tickers:
                    st.warning(f"⚠️ El ticker {ticker} ya existe en tu cartera. Se añadirá como posición adicional.")
                
                # Create new asset entry
                new_row = {
                    'Ticker': ticker,
                    'Purchase Price': purchase_price,
                    'Quantity': quantity,
                    'Asset Type': asset_type,
                }
                
                try:
                    if df is None or df.empty:
                        df = pd.DataFrame([new_row])
                    else:
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    
                    # Save the updated portfolio (overwrite mode)
                    putils.save_portfolio(username, df, overwrite=True)
                    
                    # Update session state
                    st.session_state.portfolio_df = df
                    st.session_state.portfolio_modified = True
                    
                    st.success(f"✅ Activo {ticker} agregado correctamente a tu cartera!")
                    
                    # Show quick preview of the addition
                    st.markdown("### 📊 Tu cartera actualizada:")
                    st.dataframe(df[['Ticker', 'Purchase Price', 'Quantity', 'Asset Type']], 
                               use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error al agregar el activo: {str(e)}")


def upload_portfolio_page() -> None:
    """
    Render the page that allows users to upload portfolio files with
    enhanced validation and user experience.
    """
    st.markdown('<h1 class="main-header">📤 Subir Cartera</h1>', unsafe_allow_html=True)
    
    username = st.session_state.username
    
    st.write("""
    Sube un archivo CSV o JSON que contenga tu cartera de inversiones. 
    El archivo debe incluir las columnas requeridas mostradas a continuación.
    """)
    
    # Show required format
    with st.expander("📋 Formato Requerido", expanded=True):
        st.write("**Columnas necesarias:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - **Ticker**: Símbolo del activo (ej: AAPL, TSLA)
            - **Purchase Price**: Precio de compra por unidad
            """)
        with col2:
            st.markdown("""
            - **Quantity**: Cantidad de unidades
            - **Asset Type**: Tipo de activo (Acción, ETF, etc.)
            """)
        
        # Show example
        example_df = pd.DataFrame({
            'Ticker': ['AAPL', 'TSLA', 'BTC-USD'],
            'Purchase Price': [150.00, 800.00, 45000.00],
            'Quantity': [10, 5, 0.1],
            'Asset Type': ['Acción', 'Acción', 'Cripto']
        })
        
        st.write("**Ejemplo:**")
        st.dataframe(example_df, use_container_width=True, hide_index=True)
    
    # File upload options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Selecciona tu archivo de cartera",
            type=["csv", "json"],
            help="Archivos soportados: CSV y JSON"
        )
    
    with col2:
        overwrite_mode = st.checkbox(
            "🔄 Sobrescribir cartera actual",
            value=False,
            help="Si está marcado, reemplazará tu cartera actual. Si no, creará una nueva."
        )
    
    if uploaded_file is not None:
        try:
            # Parse file based on extension
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Archivo CSV '{uploaded_file.name}' leído correctamente")
            else:  # JSON
                df = pd.read_json(uploaded_file)
                st.success(f"✅ Archivo JSON '{uploaded_file.name}' leído correctamente")
            
            # Validate required columns
            required_cols = {'Ticker', 'Purchase Price', 'Quantity', 'Asset Type'}
            missing_cols = required_cols - set(df.columns)
            
            if missing_cols:
                st.error(f"❌ Faltan las siguientes columnas requeridas: {', '.join(missing_cols)}")
                return
            
            # Data cleaning and validation
            original_rows = len(df)
            
            # Clean and standardize data
            df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
            df['Purchase Price'] = pd.to_numeric(df['Purchase Price'], errors='coerce')
            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
            df['Asset Type'] = df['Asset Type'].astype(str)
            
            # Remove invalid rows
            df = df.dropna(subset=['Ticker', 'Purchase Price', 'Quantity'])
            df = df[df['Purchase Price'] > 0]
            df = df[df['Quantity'] > 0]
            df = df[df['Ticker'].str.len() > 0]
            
            cleaned_rows = len(df)
            removed_rows = original_rows - cleaned_rows
            
            if removed_rows > 0:
                st.warning(f"⚠️ Se eliminaron {removed_rows} filas con datos inválidos")
            
            if df.empty:
                st.error("❌ No quedan datos válidos después de la limpieza")
                return
            
            # Show preview
            st.subheader("👀 Vista Previa de Datos")
            st.dataframe(df, use_container_width=True, height=300)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Activos", len(df))
            with col2:
                total_cost = (df['Purchase Price'] * df['Quantity']).sum()
                st.metric("💰 Costo Total", f"${total_cost:,.2f}")
            with col3:
                unique_types = df['Asset Type'].nunique()
                st.metric("🎯 Tipos de Activos", unique_types)
            
            # Confirmation section
            if st.button("💾 Confirmar y Guardar Cartera", type="primary"):
                try:
                    # Save portfolio
                    putils.save_portfolio(username, df, overwrite=overwrite_mode)
                    
                    # Update session state
                    load_and_set_portfolio(username)
                    
                    action = "actualizada" if overwrite_mode else "guardada como nueva"
                    st.success(f"🎉 ¡Cartera {action} exitosamente! ({len(df)} activos)")
                    
                    # Show final summary
                    st.balloons()
                    
                except Exception as save_error:
                    st.error(f"❌ Error al guardar la cartera: {str(save_error)}")
        
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")
            st.write("💡 **Sugerencias:**")
            st.write("- Verifica que el archivo esté en formato CSV o JSON válido")
            st.write("- Asegúrate de que las columnas tengan los nombres exactos requeridos")
            st.write("- Revisa que no haya caracteres especiales en los datos")


def history_page() -> None:
    """
    Enhanced portfolio history page with better file management.
    """
    st.markdown('<h1 class="main-header">📚 Historial de Carteras</h1>', unsafe_allow_html=True)
    
    username = st.session_state.username
    files = putils.list_portfolios(username)
    
    if not files:
        st.info("📝 No tienes carteras guardadas aún.")
        st.markdown("""
        ### 🚀 ¿Cómo empezar?
        1. **Añade activos** manualmente desde la página "Añadir Activo"
        2. **Sube una cartera** desde la página "Subir Cartera"
        3. Tus carteras aparecerán aquí automáticamente
        """)
        return
    
    st.write(f"📊 Tienes **{len(files)}** carteras guardadas:")
    
    # Enhanced file selection and management
    selected_file = st.selectbox(
        "🗂️ Selecciona una cartera:",
        files,
        format_func=lambda x: f"{x} {'📍 (Actual)' if x == st.session_state.selected_portfolio_file else ''}",
        help="Selecciona una cartera para cargar o gestionar"
    )
    
    if selected_file:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📂 Cargar Cartera", type="primary"):
                load_and_set_portfolio(username, selected_file)
                st.success(f"✅ Cartera '{selected_file}' cargada correctamente")
        
        with col2:
            # Download selected file
            try:
                file_path = os.path.join(putils.PORTFOLIO_DIR, selected_file)
                with open(file_path, 'rb') as f:
                    data_bytes = f.read()
                
                mime = 'application/json' if selected_file.endswith('.json') else 'text/csv'
                
                st.download_button(
                    label="💾 Descargar",
                    data=data_bytes,
                    file_name=selected_file,
                    mime=mime,
                    help="Descargar archivo seleccionado"
                )
            except Exception as e:
                st.error(f"Error al preparar descarga: {e}")
        
        with col3:
            # Delete file option
            if st.button("🗑️ Eliminar", help="Eliminar cartera seleccionada"):
                if selected_file == st.session_state.selected_portfolio_file:
                    st.error("❌ No puedes eliminar la cartera actualmente cargada")
                else:
                    try:
                        file_path = os.path.join(putils.PORTFOLIO_DIR, selected_file)
                        os.remove(file_path)
                        st.success(f"✅ Cartera '{selected_file}' eliminada")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error al eliminar: {e}")
        
        # Show file details
        try:
            file_path = os.path.join(putils.PORTFOLIO_DIR, selected_file)
            if os.path.exists(file_path):
                file_stats = os.stat(file_path)
                file_size = file_stats.st_size
                file_modified = datetime.fromtimestamp(file_stats.st_mtime)
                
                st.markdown("---")
                st.subheader("📄 Detalles del Archivo")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Tamaño", f"{file_size} bytes")
                with col2:
                    st.metric("📅 Modificado", file_modified.strftime("%Y-%m-%d"))
                with col3:
                    st.metric("🕐 Hora", file_modified.strftime("%H:%M:%S"))
                
                # Preview file content
                with st.expander("👀 Vista Previa del Contenido"):
                    try:
                        preview_df = pd.read_csv(file_path) if selected_file.endswith('.csv') else pd.read_json(file_path)
                        st.dataframe(preview_df, use_container_width=True, height=200)
                        
                        # Quick stats
                        if not preview_df.empty:
                            total_cost = (preview_df['Purchase Price'] * preview_df['Quantity']).sum()
                            st.info(f"💰 Valor de costo total: ${total_cost:,.2f} | 📊 {len(preview_df)} activos")
                    except Exception as preview_error:
                        st.error(f"Error al mostrar vista previa: {preview_error}")
                        
        except Exception as detail_error:
            st.error(f"Error al obtener detalles: {detail_error}")
    
    # Portfolio files table
    st.markdown("---")
    st.subheader("📋 Lista Completa de Carteras")
    
    if files:
        # Create a summary table
        file_data = []
        for f in files:
            try:
                file_path = os.path.join(putils.PORTFOLIO_DIR, f)
                stats = os.stat(file_path)
                size = stats.st_size
                modified = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M")
                
                # Try to get asset count
                try:
                    temp_df = pd.read_csv(file_path) if f.endswith('.csv') else pd.read_json(file_path)
                    asset_count = len(temp_df)
                except:
                    asset_count = "N/A"
                
                file_data.append({
                    'Archivo': f,
                    'Activos': asset_count,
                    'Tamaño (bytes)': size,
                    'Última Modificación': modified,
                    'Estado': '📍 Actual' if f == st.session_state.selected_portfolio_file else '📁 Guardada'
                })
            except:
                continue
        
        if file_data:
            files_df = pd.DataFrame(file_data)
            st.dataframe(files_df, use_container_width=True, hide_index=True)


def logout() -> None:
    """Log the current user out and clear session state."""
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Reinitialize with defaults
    initialize_session_state()
    
    st.success("👋 Has cerrado sesión correctamente")
    st.rerun()


# -----------------------------------------------------------------------------
# Main Application Logic
# -----------------------------------------------------------------------------

def main() -> None:
    """Main application entry point with enhanced UI and error handling."""
    
    # Main title with custom styling
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: #1f77b4;">📊 Gestor de Cartera Financiera</h1>
        <p style="color: #666; font-size: 1.1em;">Gestiona tus inversiones con análisis en tiempo real</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.authenticated:
        # Unauthenticated user: show login/registration
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["🔐 Iniciar Sesión", "📝 Registrarse"])
        
        with tab1:
            st.markdown("### 🔐 Acceso a tu Cuenta")
            
            with st.form("login_form"):
                username_input = st.text_input(
                    "👤 Usuario",
                    placeholder="Ingresa tu nombre de usuario"
                )
                password_input = st.text_input(
                    "🔒 Contraseña",
                    type="password",
                    placeholder="Ingresa tu contraseña"
                )
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    submitted = st.form_submit_button("🚀 Iniciar Sesión", type="primary")
                
                if submitted:
                    if not username_input.strip():
                        st.error("❌ El nombre de usuario no puede estar vacío")
                    elif not password_input:
                        st.error("❌ La contraseña no puede estar vacía")
                    else:
                        with st.spinner("🔍 Verificando credenciales..."):
                            if authenticate_user(username_input.strip(), password_input):
                                st.session_state.authenticated = True
                                st.session_state.username = username_input.strip()
                                st.session_state.first_login = True
                                
                                # Load user's latest portfolio
                                load_and_set_portfolio(username_input.strip())
                                
                                st.success("✅ ¡Inicio de sesión exitoso!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Usuario o contraseña incorrectos")
        
        with tab2:
            st.markdown("### 📝 Crear Nueva Cuenta")
            
            with st.form("register_form"):
                new_username = st.text_input(
                    "👤 Nombre de Usuario",
                    placeholder="Elige un nombre de usuario único"
                )
                new_password = st.text_input(
                    "🔒 Contraseña",
                    type="password",
                    placeholder="Mínimo 6 caracteres"
                )
                confirm_password = st.text_input(
                    "🔒 Confirmar Contraseña",
                    type="password",
                    placeholder="Repite tu contraseña"
                )
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    submitted_reg = st.form_submit_button("✨ Crear Cuenta", type="primary")
                
                if submitted_reg:
                    # Validation
                    username_clean = new_username.strip()
                    
                    if not username_clean:
                        st.error("❌ El nombre de usuario no puede estar vacío")
                    elif len(username_clean) < 3:
                        st.error("❌ El nombre de usuario debe tener al menos 3 caracteres")
                    elif len(new_password) < 6:
                        st.error("❌ La contraseña debe tener al menos 6 caracteres")
                    elif new_password != confirm_password:
                        st.error("❌ Las contraseñas no coinciden")
                    else:
                        with st.spinner("👤 Creando cuenta..."):
                            if register_user(username_clean, new_password):
                                st.success("🎉 ¡Usuario registrado correctamente! Ahora puedes iniciar sesión.")
                                st.balloons()
                            else:
                                st.error("❌ El nombre de usuario ya existe. Prueba con otro.")
        
        # Footer for unauthenticated users
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>💡 <strong>¿Primera vez aquí?</strong> Crea una cuenta para empezar a gestionar tus inversiones</p>
            <p>🔒 Tus datos están seguros con cifrado PBKDF2</p>
        </div>
        """, unsafe_allow_html=True)
        
        return

    # Authenticated user interface
    show_welcome_message()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-info">
            <h3>👤 Usuario Activo</h3>
            <p><strong>{st.session_state.username}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Current portfolio info
        if st.session_state.portfolio_df is not None and not st.session_state.portfolio_df.empty:
            asset_count = len(st.session_state.portfolio_df)
            st.markdown(f"""
            <div style="background-color: #e7f3ff; padding: 0.5rem; border-radius: 0.25rem; margin-bottom: 1rem;">
                <small>📊 <strong>Cartera Actual:</strong><br>
                {asset_count} activos cargados</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Navigation menu
        st.markdown("### 🧭 Navegación")
        page = st.radio(
            "Selecciona una página:",
            [
                "📊 Resumen de Cartera",
                "➕ Añadir Activo", 
                "📤 Subir Cartera",
                "📚 Historial de Carteras",
                "🚪 Cerrar Sesión"
            ],
            label_visibility="collapsed"
        )
        
        # Quick actions
        st.markdown("---")
        st.markdown("### ⚡ Acciones Rápidas")
        
        if st.button("🔄 Actualizar Precios", help="Refrescar precios de mercado"):
            st.session_state.price_cache = {}
            st.session_state.price_cache_time = 0
            st.success("✅ Cache de precios limpiado")
            st.rerun()
        
        if st.session_state.portfolio_df is not None and not st.session_state.portfolio_df.empty:
            if st.button("📊 Vista Rápida", help="Mostrar resumen rápido"):
                with st.expander("📈 Resumen Rápido", expanded=True):
                    df = st.session_state.portfolio_df
                    st.write(f"**Activos:** {len(df)}")
                    st.write(f"**Tipos:** {df['Asset Type'].nunique()}")
                    total_cost = (df['Purchase Price'] * df['Quantity']).sum()
                    st.write(f"**Costo Total:** ${total_cost:,.2f}")

    # Main content routing
    if page == "📊 Resumen de Cartera":
        display_portfolio_overview()
    elif page == "➕ Añadir Activo":
        add_asset_page()
    elif page == "📤 Subir Cartera":
        upload_portfolio_page()
    elif page == "📚 Historial de Carteras":
        history_page()
    elif page == "🚪 Cerrar Sesión":
        if st.button("⚠️ Confirmar Cierre de Sesión", type="secondary"):
            logout()
        else:
            st.info("👆 Haz clic en el botón de arriba para confirmar el cierre de sesión")


if __name__ == "__main__":
    main()
