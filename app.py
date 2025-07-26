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

import pandas as pd
import numpy as np
import streamlit as st

from auth import authenticate_user, register_user
import portfolio_utils as putils


# -----------------------------------------------------------------------------
# Session state initialisation
#
# Streamlit runs scripts from top to bottom on each interaction.  To maintain
# user state between interactions (such as login status and loaded portfolio),
# we persist values in st.session_state.
# -----------------------------------------------------------------------------

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = None
if 'selected_portfolio_file' not in st.session_state:
    st.session_state.selected_portfolio_file = None


# -----------------------------------------------------------------------------
# Helper functions for the Streamlit UI
# -----------------------------------------------------------------------------

def load_and_set_portfolio(username: str, filename: Optional[str] = None) -> None:
    """Load a portfolio from disk and store it in session state."""
    df = putils.load_portfolio(username, filename)
    st.session_state.portfolio_df = df
    st.session_state.selected_portfolio_file = filename


def display_portfolio_overview() -> None:
    """
    Render the main portfolio overview page.

    Displays the list of available portfolios to load, shows current
    portfolio metrics, visualises the composition, identifies top and
    worst performers, and provides insights such as diversification
    suggestions.  Allows the user to download the current portfolio.
    """
    username = st.session_state.username
    st.header("📊 Resumen de Cartera")
    # Provide a dropdown to select among saved portfolios
    portfolios = putils.list_portfolios(username)
    if portfolios:
        default_index = 0
        if st.session_state.selected_portfolio_file is not None:
            # If a portfolio is already selected, set it as default
            try:
                default_index = portfolios.index(st.session_state.selected_portfolio_file)
            except ValueError:
                default_index = 0
        selected_file = st.selectbox(
            "Selecciona una cartera para cargar",
            portfolios,
            index=default_index,
            key="portfolio_select_box",
        )
        if selected_file != st.session_state.selected_portfolio_file:
            load_and_set_portfolio(username, selected_file)
    else:
        st.info("Aún no tienes ninguna cartera guardada. Sube una o crea una nueva desde la barra lateral.")

    df = st.session_state.portfolio_df
    if df is None or df.empty:
        return
    # Fetch current prices and compute metrics
    tickers = df['Ticker'].tolist()
    price_dict = putils.fetch_current_prices(tickers)
    metrics_df = putils.compute_metrics(df, price_dict)
    # Compute RSI and volatility for each asset
    rsi_list = []
    vol_list = []
    for ticker in metrics_df['Ticker']:
        rsi = np.nan
        vol = np.nan
        if putils.yf is not None:
            try:
                hist = putils.yf.Ticker(ticker).history(period="6mo")['Close']
                rsi = putils.compute_rsi(hist)
                vol = putils.compute_volatility(hist)
            except Exception:
                rsi = np.nan
                vol = np.nan
        rsi_list.append(rsi)
        vol_list.append(vol)
    metrics_df['RSI'] = rsi_list
    metrics_df['Volatilidad (%)'] = vol_list

    # Display metrics table
    st.subheader("Detalle de activos")
    # Format numbers for display
    display_df = metrics_df.copy()
    display_df['Purchase Price'] = display_df['Purchase Price'].map('{:,.2f}'.format)
    display_df['Current Price'] = display_df['Current Price'].map(lambda x: '{:,.2f}'.format(x) if pd.notna(x) else 'N/A')
    display_df['Total Value'] = display_df['Total Value'].map(lambda x: '{:,.2f}'.format(x) if pd.notna(x) else 'N/A')
    display_df['P/L'] = display_df['P/L'].map(lambda x: '{:,.2f}'.format(x) if pd.notna(x) else 'N/A')
    display_df['P/L %'] = display_df['P/L %'].map(lambda x: '{:.2f}%'.format(x) if pd.notna(x) else 'N/A')
    display_df['Weight %'] = display_df['Weight %'].map(lambda x: '{:.2f}%'.format(x) if pd.notna(x) else 'N/A')
    display_df['RSI'] = display_df['RSI'].map(lambda x: '{:.2f}'.format(x) if pd.notna(x) else 'N/A')
    display_df['Volatilidad (%)'] = display_df['Volatilidad (%)'].map(lambda x: '{:.2f}%'.format(x) if pd.notna(x) else 'N/A')
    st.dataframe(display_df, use_container_width=True)

    # Summary statistics
    total_value = metrics_df['Total Value'].sum()
    total_cost = metrics_df['Cost Basis'].sum()
    total_pl = total_value - total_cost
    total_pl_pct = (total_pl / total_cost * 100.0) if total_cost > 0 else np.nan
    col1, col2, col3 = st.columns(3)
    col1.metric("Valor total de la cartera", f"{total_value:,.2f}")
    col2.metric("Ganancia/Pérdida total", f"{total_pl:,.2f}", f"{total_pl_pct:.2f}%" if pd.notna(total_pl_pct) else "")
    col3.metric("Número de activos", len(metrics_df))

    # Top and worst performers
    top_df, worst_df = putils.top_and_worst_assets(metrics_df)
    with st.expander("Top y Peor rendimiento"):
        col_top, col_worst = st.columns(2)
        with col_top:
            st.write("### Top activos")
            if not top_df.empty:
                st.dataframe(top_df[['Ticker','P/L %','P/L','Weight %']], use_container_width=True)
            else:
                st.write("Sin suficientes datos")
        with col_worst:
            st.write("### Peor rendimiento")
            if not worst_df.empty:
                st.dataframe(worst_df[['Ticker','P/L %','P/L','Weight %']], use_container_width=True)
            else:
                st.write("Sin suficientes datos")

    # Asset allocation breakdown
    breakdown_df = putils.asset_breakdown(metrics_df)
    if not breakdown_df.empty and putils.px is not None:
        fig = putils.px.pie(
            breakdown_df,
            names='Asset Type',
            values='Total Value',
            title='Distribución por tipo de activo',
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    # Diversification suggestion
    suggestion = putils.suggest_diversification(metrics_df)
    if suggestion:
        st.warning(suggestion)

    # Download current portfolio
    with st.expander("Descargar cartera actual"):
        csv_data = metrics_df.to_csv(index=False).encode('utf-8')
        json_data = metrics_df.to_json(orient="records", indent=2).encode('utf-8')
        st.download_button(
            label="Descargar CSV",
            data=csv_data,
            file_name=f"{username}_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
        )
        st.download_button(
            label="Descargar JSON",
            data=json_data,
            file_name=f"{username}_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime='application/json',
        )


def add_asset_page() -> None:
    """
    Render the page that allows the user to add a new asset to the current
    portfolio.  The new asset is immediately saved to disk and the current
    session state is updated.
    """
    st.header("➕ Añadir Activo")
    username = st.session_state.username
    df = st.session_state.portfolio_df
    with st.form("add_asset_form"):
        ticker = st.text_input("Ticker", max_chars=20).strip().upper()
        purchase_price = st.number_input("Precio de compra", min_value=0.0, format="%0.2f")
        quantity = st.number_input("Cantidad", min_value=0.0, format="%0.4f")
        asset_type = st.selectbox(
            "Tipo de activo",
            ["Acción", "ETF", "Cripto", "Oro", "Bono", "Otro"],
            index=0,
        )
        submitted = st.form_submit_button("Agregar a cartera")
        if submitted:
            if ticker == "" or quantity <= 0:
                st.error("Debes proporcionar un ticker válido y una cantidad positiva.")
            else:
                new_row = {
                    'Ticker': ticker,
                    'Purchase Price': purchase_price,
                    'Quantity': quantity,
                    'Asset Type': asset_type,
                }
                if df is None or df.empty:
                    df = pd.DataFrame([new_row])
                else:
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                # Save the updated portfolio
                putils.save_portfolio(username, df)
                # Reload the latest portfolio and update session
                load_and_set_portfolio(username)
                st.success(f"Activo {ticker} agregado correctamente.")


def upload_portfolio_page() -> None:
    """
    Render the page that allows the user to upload a portfolio file.  The
    uploaded file must be in CSV or JSON format with the required columns.
    """
    st.header("📤 Subir Cartera")
    username = st.session_state.username
    st.write("Puedes subir un archivo CSV o JSON que contenga tus activos. "
             "Las columnas necesarias son: 'Ticker', 'Purchase Price', 'Quantity' y 'Asset Type'.")
    uploaded_file = st.file_uploader("Elige un archivo", type=["csv", "json"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
            required_cols = {'Ticker', 'Purchase Price', 'Quantity', 'Asset Type'}
            if not required_cols.issubset(df.columns):
                st.error(f"El archivo debe contener las columnas {required_cols}.")
                return
            # Standardise column ordering and types
            df['Ticker'] = df['Ticker'].astype(str).str.upper()
            df['Purchase Price'] = pd.to_numeric(df['Purchase Price'], errors='coerce')
            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
            df['Asset Type'] = df['Asset Type'].astype(str)
            df = df.dropna(subset=['Ticker','Purchase Price','Quantity'])
            # Save portfolio
            putils.save_portfolio(username, df)
            load_and_set_portfolio(username)
            st.success("Cartera subida y guardada con éxito.")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")


def history_page() -> None:
    """
    Render the page showing the history of saved portfolios for the user.  The
    user can select a portfolio from the table to load it into the current
    session or download it.
    """
    st.header("📚 Historial de Carteras")
    username = st.session_state.username
    files = putils.list_portfolios(username)
    if not files:
        st.info("No tienes carteras guardadas.")
        return
    selected = st.selectbox("Selecciona un archivo para cargar", files)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cargar cartera seleccionada"):
            load_and_set_portfolio(username, selected)
            st.success(f"Cartera {selected} cargada.")
    with col2:
        file_path = os.path.join(putils.PORTFOLIO_DIR, selected)
        with open(file_path, 'rb') as f:
            data_bytes = f.read()
        mime = 'application/json' if selected.endswith('.json') else 'text/csv'
        st.download_button(
            label="Descargar archivo",
            data=data_bytes,
            file_name=selected,
            mime=mime,
        )
    # Display list of portfolios in a table
    table_df = pd.DataFrame({'Archivo': files})
    st.dataframe(table_df, use_container_width=True)


def logout() -> None:
    """Log the current user out and clear session state."""
    st.session_state.authenticated = False
    st.session_state.username = ''
    st.session_state.portfolio_df = None
    st.session_state.selected_portfolio_file = None


# -----------------------------------------------------------------------------
# Main Application Logic
# -----------------------------------------------------------------------------

def main() -> None:
    st.title("Gestor de Cartera Financiera")
    if not st.session_state.authenticated:
        # Unauthenticated user: show login/registration tabs
        tab1, tab2 = st.tabs(["Iniciar sesión", "Registrarse"])
        with tab1:
            st.subheader("Iniciar sesión")
            with st.form("login_form"):
                username_input = st.text_input("Usuario")
                password_input = st.text_input("Contraseña", type="password")
                submitted = st.form_submit_button("Iniciar sesión")
                if submitted:
                    if authenticate_user(username_input, password_input):
                        st.session_state.authenticated = True
                        st.session_state.username = username_input
                        load_and_set_portfolio(username_input)
                        st.success("Inicio de sesión exitoso.")
                    else:
                        st.error("Usuario o contraseña incorrectos.")
        with tab2:
            st.subheader("Registrarse")
            with st.form("register_form"):
                new_username = st.text_input("Nombre de usuario")
                new_password = st.text_input("Contraseña", type="password")
                confirm_password = st.text_input("Confirmar contraseña", type="password")
                submitted_reg = st.form_submit_button("Registrar")
                if submitted_reg:
                    if new_username.strip() == "":
                        st.error("El usuario no puede estar vacío.")
                    elif new_password != confirm_password:
                        st.error("Las contraseñas no coinciden.")
                    else:
                        if register_user(new_username.strip(), new_password):
                            st.success("Usuario registrado correctamente. Ahora puedes iniciar sesión.")
                        else:
                            st.error("El nombre de usuario ya existe.")
        # End of unauthenticated UI
        return

    # Authenticated user: show navigation sidebar and content
    st.sidebar.write(f"👤 Sesión iniciada como **{st.session_state.username}**")
    page = st.sidebar.radio(
        "Navegación",
        (
            "Resumen de Cartera",
            "Añadir Activo",
            "Subir Cartera",
            "Historial de Carteras",
            "Cerrar Sesión",
        ),
    )
    if page == "Resumen de Cartera":
        display_portfolio_overview()
    elif page == "Añadir Activo":
        add_asset_page()
    elif page == "Subir Cartera":
        upload_portfolio_page()
    elif page == "Historial de Carteras":
        history_page()
    elif page == "Cerrar Sesión":
        logout()
        st.success("Has cerrado sesión.")


if __name__ == "__main__":
    main()